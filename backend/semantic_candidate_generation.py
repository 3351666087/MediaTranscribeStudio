"""Bounded challenger execution for semantic candidate-generation requests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .persistence import canonical_json_sha256, validate_strict_json
from .asr_evidence import validate_asr_candidate_set
from .semantic_candidate_lattice import (
    extend_semantic_candidate_lattice,
    semantic_candidate_payload_sha256,
    validate_semantic_candidate_lattice,
)
from .semantic_composition import (
    SemanticCompositionError,
    validate_semantic_job_arbitration,
)
from .voice_activity import validate_voice_activity


SEMANTIC_CANDIDATE_GENERATION_SCHEMA_VERSION = "1.0.0"
SEMANTIC_CANDIDATE_GENERATION_ARTIFACT_TYPE = "semantic-candidate-generation"

SemanticCandidateGenerator = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


def build_timeline_challenger_result(
    *,
    turns: Sequence[Mapping[str, Any]],
    source_duration_ms: int,
    system_id: str,
    revision: str,
    artifact_sha256: str,
    model_manifest_sha256: str | None,
    local_speaker_field: str = "localSpeaker",
) -> dict[str, Any]:
    """Normalize a registered diarizer's local labels into one timeline candidate."""

    if (
        isinstance(source_duration_ms, bool)
        or not isinstance(source_duration_ms, int)
        or source_duration_ms < 1
    ):
        raise _fail("timeline challenger source duration is invalid")
    if (
        not isinstance(turns, Sequence)
        or isinstance(turns, (str, bytes, bytearray))
        or not turns
        or len(turns) > 10_000
    ):
        raise _fail("timeline challenger turns are missing or exceed 10000")
    speaker_field = _text(
        local_speaker_field,
        "localSpeakerField",
        maximum=80,
    )
    normalized: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise _fail(f"timeline challenger turns[{index}] must be an object")
        start = turn.get("startMs")
        end = turn.get("endMs")
        label = turn.get(speaker_field)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > source_duration_ms
        ):
            raise _fail(f"timeline challenger turns[{index}] timing is invalid")
        normalized.append(
            {
                "startMs": start,
                "endMs": end,
                "localSpeaker": _text(
                    label,
                    f"timeline challenger turns[{index}].{speaker_field}",
                    maximum=120,
                ),
                **(
                    {
                        "text": _text(
                            turn.get("text"),
                            f"timeline challenger turns[{index}].text",
                            maximum=20_000,
                        )
                    }
                    if "text" in turn
                    else {}
                ),
            }
        )
    local_speakers = sorted(
        {turn["localSpeaker"] for turn in normalized}
    )
    mapping = {
        local: f"speaker-{index}"
        for index, local in enumerate(local_speakers, start=1)
    }
    normalized.sort(
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["localSpeaker"],
        )
    )
    candidate_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(normalized):
        overlap = any(
            other_index != index
            and other["localSpeaker"] != turn["localSpeaker"]
            and max(turn["startMs"], other["startMs"])
            < min(turn["endMs"], other["endMs"])
            for other_index, other in enumerate(normalized)
        )
        candidate_turn = {
            "startMs": turn["startMs"],
            "endMs": turn["endMs"],
            "speakerId": mapping[turn["localSpeaker"]],
            "overlap": overlap,
        }
        if "text" in turn:
            candidate_turn["text"] = turn["text"]
        candidate_turns.append(candidate_turn)
    producer = {
        "producerType": "model",
        "systemId": _text(system_id, "systemId", maximum=160),
        "revision": _text(revision, "revision", maximum=160),
        "artifactSha256": artifact_sha256,
        "modelManifestSha256": model_manifest_sha256,
        "identityStatus": (
            "manifest-bound"
            if model_manifest_sha256 is not None
            else "artifact-bound"
        ),
    }
    return {
        "producer": producer,
        "candidates": [
            {
                "payload": {
                    "speakerCount": len(local_speakers),
                    "speakerIds": [
                        f"speaker-{index}"
                        for index in range(1, len(local_speakers) + 1)
                    ],
                    "timelineKind": "challenger",
                    "startMs": 0,
                    "endMs": source_duration_ms,
                    "turns": candidate_turns,
                }
            }
        ],
    }


def build_voice_activity_challenger_result(
    *,
    voice_activity: Mapping[str, Any],
    artifact_sha256: str,
) -> dict[str, Any]:
    """Bind canonical voice-activity evidence to one speech disposition."""

    evidence = validate_voice_activity(voice_activity)
    classification = (
        "transcribable-speech"
        if evidence["hasTranscribableSpeech"]
        else "no-transcribable-speech"
    )
    return {
        "producer": {
            "producerType": "deterministic",
            "systemId": _text(
                evidence["provider"]["id"],
                "voiceActivity.provider.id",
                maximum=160,
            ),
            "revision": _text(
                evidence["provider"]["version"],
                "voiceActivity.provider.version",
                maximum=160,
            ),
            "artifactSha256": artifact_sha256,
            "modelManifestSha256": None,
            "identityStatus": "artifact-bound",
        },
        "candidates": [
            {
                "payload": {
                    "classification": classification,
                    "startMs": 0,
                    "endMs": evidence["mediaDurationMs"],
                    "speechDurationMs": evidence["speechDurationMs"],
                    "speechRatio": evidence["speechRatio"],
                    "speechWindowCount": evidence["speechWindowCount"],
                }
            }
        ],
    }


def build_open_set_lid_challenger_result(
    *,
    segment_id: str,
    start_ms: int,
    end_ms: int,
    language: str,
    confidence: float | None,
    system_id: str,
    revision: str,
    artifact_sha256: str,
    model_manifest_sha256: str | None,
    evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind one independently produced open-set LID decision."""

    return {
        "producer": {
            "producerType": "model",
            "systemId": _text(system_id, "systemId", maximum=160),
            "revision": _text(revision, "revision", maximum=160),
            "artifactSha256": artifact_sha256,
            "modelManifestSha256": model_manifest_sha256,
            "identityStatus": (
                "manifest-bound"
                if model_manifest_sha256 is not None
                else "artifact-bound"
            ),
        },
        "candidates": [
            {
                "payload": {
                    "segmentId": _text(
                        segment_id,
                        "segmentId",
                        maximum=160,
                    ),
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "language": _text(language, "language", maximum=255),
                    "confidence": confidence,
                    **(
                        {"evidenceSha256": evidence_sha256}
                        if evidence_sha256 is not None
                        else {}
                    ),
                }
            }
        ],
    }


def build_asr_text_challenger_result(
    *,
    segment_id: str,
    start_ms: int,
    end_ms: int,
    candidate_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every eligible result from one provider-native ASR re-decode."""

    validated = validate_asr_candidate_set(
        candidate_set,
        expected_start_ms=start_ms,
        expected_end_ms=end_ms,
    )
    producer = {
        "producerType": "model",
        "systemId": validated["modelId"],
        "revision": validated["modelRevision"],
        "artifactSha256": validated["candidateSetSha256"],
        "modelManifestSha256": validated["modelManifestSha256"],
        "identityStatus": validated["modelIdentityStatus"],
    }
    candidates = [
        {
            "payload": {
                "segmentId": _text(segment_id, "segmentId", maximum=160),
                "startMs": start_ms,
                "endMs": end_ms,
                "text": item["text"],
                "language": item["language"],
                "sourceCandidateId": item["candidateId"],
                "candidateSetSha256": validated["candidateSetSha256"],
            }
        }
        for item in validated["nBest"]
        if item["lexicalRepairEligible"] is True or int(item["rank"]) == 1
    ]
    return {"producer": producer, "candidates": candidates}


def _fail(message: str) -> SemanticCompositionError:
    return SemanticCompositionError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any, field: str, *, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise _fail(f"{field} exceeds {maximum} characters")
    return result


def _group_index(lattice: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(group["groupId"]): {
            **group,
            "domain": str(domain["domain"]),
        }
        for domain in lattice["domains"]
        for group in domain["groups"]
    }


def _normalize_generator_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    raw_candidate_results: list[dict[str, Any]] = []
    if set(value) == {"producer", "candidates"}:
        producer = value.get("producer")
        raw_candidates = value.get("candidates")
        if not isinstance(producer, Mapping) or not isinstance(
            raw_candidates,
            list,
        ):
            raise _fail(f"{field} producer or candidates are invalid")
        raw_candidate_results = [
            {"producer": dict(producer), **dict(candidate)}
            for candidate in raw_candidates
            if isinstance(candidate, Mapping)
        ]
        if len(raw_candidate_results) != len(raw_candidates):
            raise _fail(f"{field}.candidates must contain objects")
    elif set(value) == {"candidateResults"}:
        raw_results = value.get("candidateResults")
        if not isinstance(raw_results, list):
            raise _fail(f"{field}.candidateResults must be an array")
        raw_candidate_results = [
            dict(result) for result in raw_results if isinstance(result, Mapping)
        ]
        if len(raw_candidate_results) != len(raw_results):
            raise _fail(f"{field}.candidateResults must contain objects")
    else:
        raise _fail(f"{field} fields do not match the generator result contract")
    if not raw_candidate_results or len(raw_candidate_results) > 8:
        raise _fail(f"{field} must contain between 1 and 8 candidate results")
    candidates: list[dict[str, Any]] = []
    first_producer: dict[str, Any] | None = None
    for index, raw in enumerate(raw_candidate_results):
        candidate_field = f"{field}.candidates[{index}]"
        if set(raw) != {"producer", "payload"} or not isinstance(
            raw.get("producer"),
            Mapping,
        ):
            raise _fail(
                f"{candidate_field} must contain producer and payload"
            )
        producer = dict(raw["producer"])
        if first_producer is None:
            first_producer = producer
        candidate = {
            "payload": raw["payload"],
            "producers": [producer],
            "selectionEligible": True,
            "eligibilityReason": "eligible",
        }
        try:
            validate_strict_json(candidate)
        except ValueError as exc:
            raise _fail(
                f"{candidate_field} must contain strict finite JSON"
            ) from exc
        candidates.append(candidate)
    supplement = {
        "domain": request["domain"],
        "groupId": request["groupId"],
        "scopeId": request["scopeId"],
        "candidates": candidates,
    }
    assert first_producer is not None
    return first_producer, [supplement]


def _fulfilled_trace(
    requests: list[Mapping[str, Any]],
    *,
    before_lattice: Mapping[str, Any],
    after_lattice: Mapping[str, Any],
    supplements: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    before_groups = _group_index(before_lattice)
    after_groups = _group_index(after_lattice)
    supplements_by_group = {
        str(item["groupId"]): item for item in supplements
    }
    fulfilled: list[dict[str, Any]] = []
    for request in requests:
        group_id = str(request["groupId"])
        before = before_groups[group_id]
        after = after_groups[group_id]
        supplement = supplements_by_group.get(group_id)
        if supplement is None:
            raise _fail(f"candidate generation omitted supplement for {group_id}")
        raw_candidates = supplement.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _fail(f"candidate generation supplement for {group_id} is empty")
        producers_by_hash: dict[str, dict[str, Any]] = {}
        for candidate in raw_candidates:
            candidate_producers = (
                candidate.get("producers")
                if isinstance(candidate, Mapping)
                else None
            )
            if (
                not isinstance(candidate_producers, list)
                or len(candidate_producers) != 1
                or not isinstance(candidate_producers[0], Mapping)
            ):
                raise _fail(
                    f"candidate generation supplement for {group_id} must "
                    "bind each candidate to exactly one producer"
                )
            normalized_producer = dict(candidate_producers[0])
            producers_by_hash[
                canonical_json_sha256(normalized_producer)
            ] = normalized_producer
        producers = [
            producers_by_hash[key] for key in sorted(producers_by_hash)
        ]
        before_ids = {
            str(candidate["candidateId"]) for candidate in before["candidates"]
        }
        generated_ids = sorted(
            str(candidate["candidateId"])
            for candidate in after["candidates"]
            if candidate["candidateId"] not in before_ids
        )
        minimum = int(request["minimumAlternativeCount"])
        if (
            not generated_ids
            or int(after["eligibleCandidateCount"]) < minimum
            or after["status"] != "available"
        ):
            raise _fail(
                f"candidate generator did not satisfy {group_id} minimum "
                "eligible alternative count"
            )
        trace = {
            "domain": request["domain"],
            "groupId": group_id,
            "scopeId": request["scopeId"],
            "requestKind": request["requestKind"],
            "minimumAlternativeCount": minimum,
            "generatedCandidateIds": generated_ids,
            "outputGroupStatus": after["status"],
            "outputEligibleCandidateCount": after[
                "eligibleCandidateCount"
            ],
        }
        if len(producers) == 1:
            trace["producer"] = producers[0]
        else:
            trace["producers"] = producers
        fulfilled.append(trace)
    fulfilled.sort(
        key=lambda item: (
            str(item["domain"]),
            str(item["scopeId"]),
            str(item["groupId"]),
        )
    )
    return fulfilled


class SemanticCandidateGenerationRegistry:
    """Execute only explicitly registered, bounded local challenger handlers."""

    def __init__(
        self,
        handlers: Mapping[str, SemanticCandidateGenerator],
    ) -> None:
        if not isinstance(handlers, Mapping) or not handlers:
            raise ValueError("semantic candidate generators must not be empty")
        normalized: dict[str, SemanticCandidateGenerator] = {}
        for request_kind, handler in handlers.items():
            kind = _text(
                request_kind,
                "candidate generator request kind",
                maximum=120,
            )
            if not callable(handler):
                raise ValueError(
                    f"semantic candidate generator {kind} must be callable"
                )
            normalized[kind] = handler
        self._handlers = normalized

    @property
    def request_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def fulfill(
        self,
        document: Mapping[str, Any],
        lattice: Mapping[str, Any],
        arbitration_artifact: Mapping[str, Any],
        *,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        """Run every requested challenger once and rebuild one extended lattice."""

        transcript_sha = canonical_json_sha256(document)
        validated_lattice = validate_semantic_candidate_lattice(
            lattice,
            expected_transcript_sha256=transcript_sha,
        )
        job_id = str(document.get("jobId") or "")
        arbitration = validate_semantic_job_arbitration(
            arbitration_artifact,
            expected_job_id=job_id,
            expected_lattice=validated_lattice,
        )
        if arbitration["status"] != "candidate-generation-required":
            raise _fail(
                "candidate generation requires an arbitration artifact with "
                "unresolved challenger requests"
            )
        requests = arbitration["candidateGenerationRequests"]
        if len(requests) > 128:
            raise _fail("candidate generation request count exceeds 128")
        supplements: list[dict[str, Any]] = []
        handled_requests: list[dict[str, Any]] = []
        unfulfilled_requests: list[dict[str, Any]] = []
        matched_handler_count = 0
        existing_payloads = {
            str(group["groupId"]): {
                str(candidate["payloadSha256"])
                for candidate in group["candidates"]
            }
            for domain in validated_lattice["domains"]
            for group in domain["groups"]
        }
        for index, request in enumerate(requests):
            group_id = request["groupId"]
            request_kind = str(request["requestKind"])
            handler = self._handlers.get(request_kind)
            if handler is None:
                unfulfilled_requests.append(dict(request))
                continue
            matched_handler_count += 1
            if group_id is None:
                raise _fail(
                    "domain-level candidate generation requires a domain "
                    "bootstrap handler and is not yet composable"
                )
            immutable_request = json.loads(
                json.dumps(request, ensure_ascii=False)
            )
            immutable_document = json.loads(
                json.dumps(document, ensure_ascii=False)
            )
            immutable_lattice = json.loads(
                json.dumps(validated_lattice, ensure_ascii=False)
            )
            raw_result = handler(
                immutable_request,
                immutable_document,
                immutable_lattice,
            )
            _, generated = _normalize_generator_result(
                raw_result,
                request=request,
                field=f"generatorResults[{index}]",
            )
            novel_supplements: list[dict[str, Any]] = []
            known_payloads = set(existing_payloads[str(group_id)])
            for supplement in generated:
                novel_candidates: list[dict[str, Any]] = []
                for candidate in supplement["candidates"]:
                    payload_sha256 = semantic_candidate_payload_sha256(
                        validated_lattice,
                        domain=str(supplement["domain"]),
                        group_id=str(supplement["groupId"]),
                        scope_id=str(supplement["scopeId"]),
                        payload=candidate["payload"],
                    )
                    if payload_sha256 in known_payloads:
                        continue
                    known_payloads.add(payload_sha256)
                    novel_candidates.append(candidate)
                if novel_candidates:
                    novel_supplements.append(
                        {
                            **supplement,
                            "candidates": novel_candidates,
                        }
                    )
            if novel_supplements:
                existing_payloads[str(group_id)] = known_payloads
                supplements.extend(novel_supplements)
                handled_requests.append(dict(request))
            else:
                unfulfilled_requests.append(dict(request))
        if not supplements and matched_handler_count == 0:
            raise _fail(
                "no registered candidate generator matches the arbitration requests"
            )

        extended = (
            extend_semantic_candidate_lattice(
                validated_lattice,
                supplemental_groups=supplements,
            )
            if supplements
            else validated_lattice
        )
        fulfilled = _fulfilled_trace(
            handled_requests,
            before_lattice=validated_lattice,
            after_lattice=extended,
            supplements=supplements,
        )
        supplements.sort(
            key=lambda item: (
                str(item["domain"]),
                str(item["scopeId"]),
                str(item["groupId"]),
            )
        )
        generated = _text(
            generated_at or _utc_now(),
            "generatedAt",
            maximum=80,
        )
        generation_body = {
            "inputLatticeSha256": validated_lattice["latticeSha256"],
            "arbitrationDecisionSha256": arbitration["decisionSha256"],
            "supplementalGroups": supplements,
            "outputLatticeSha256": extended["latticeSha256"],
        }
        generation_sha = canonical_json_sha256(generation_body)
        artifact = {
            "schemaVersion": SEMANTIC_CANDIDATE_GENERATION_SCHEMA_VERSION,
            "artifactType": SEMANTIC_CANDIDATE_GENERATION_ARTIFACT_TYPE,
            "artifactId": "semantic-generation-" + generation_sha[:24],
            "generationSha256": generation_sha,
            "jobId": job_id,
            "generatedAt": generated,
            "binding": {
                "sourceMediaSha256": validated_lattice["binding"][
                    "sourceMediaSha256"
                ],
                "transcriptSha256": transcript_sha,
                "inputLatticeSha256": validated_lattice["latticeSha256"],
                "arbitrationArtifactSha256": canonical_json_sha256(
                    arbitration
                ),
                "arbitrationDecisionSha256": arbitration["decisionSha256"],
                "outputLatticeSha256": extended["latticeSha256"],
            },
            "status": (
                "completed"
                if not unfulfilled_requests
                else "partial"
            ),
            "fulfilledRequests": fulfilled,
            "unfulfilledRequests": unfulfilled_requests,
            "supplementalGroups": supplements,
            "outputLattice": extended,
            "metrics": {
                "requestCount": len(requests),
                "fulfilledRequestCount": len(fulfilled),
                "unfulfilledRequestCount": len(unfulfilled_requests),
                "generatedCandidateCount": sum(
                    len(item["generatedCandidateIds"]) for item in fulfilled
                ),
                "inputAvailableGroupCount": validated_lattice[
                    "availability"
                ]["availableGroupCount"],
                "outputAvailableGroupCount": extended["availability"][
                    "availableGroupCount"
                ],
            },
        }
        validate_strict_json(artifact)
        return validate_semantic_candidate_generation(
            artifact,
            expected_document=document,
            expected_lattice=validated_lattice,
            expected_arbitration=arbitration,
        )


def validate_semantic_candidate_generation(
    artifact: Mapping[str, Any],
    *,
    expected_document: Mapping[str, Any],
    expected_lattice: Mapping[str, Any],
    expected_arbitration: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the extended lattice and reject generated-candidate tampering."""

    value = dict(artifact)
    try:
        validate_strict_json(value)
    except ValueError as exc:
        raise _fail(
            "semantic candidate generation must contain strict finite JSON"
        ) from exc
    required = {
        "schemaVersion",
        "artifactType",
        "artifactId",
        "generationSha256",
        "jobId",
        "generatedAt",
        "binding",
        "status",
        "fulfilledRequests",
        "unfulfilledRequests",
        "supplementalGroups",
        "outputLattice",
        "metrics",
    }
    if set(value) != required:
        raise _fail("semantic candidate generation fields do not match schema 1.0.0")
    lattice = validate_semantic_candidate_lattice(
        expected_lattice,
        expected_transcript_sha256=canonical_json_sha256(expected_document),
    )
    job_id = str(expected_document.get("jobId") or "")
    arbitration = validate_semantic_job_arbitration(
        expected_arbitration,
        expected_job_id=job_id,
        expected_lattice=lattice,
    )
    if (
        value.get("schemaVersion")
        != SEMANTIC_CANDIDATE_GENERATION_SCHEMA_VERSION
        or value.get("artifactType")
        != SEMANTIC_CANDIDATE_GENERATION_ARTIFACT_TYPE
        or value.get("jobId") != job_id
        or value.get("status") not in {"completed", "partial"}
        or not isinstance(value.get("generatedAt"), str)
        or not value["generatedAt"]
    ):
        raise _fail("semantic candidate generation identity is invalid")
    supplements = value.get("supplementalGroups")
    if not isinstance(supplements, list):
        raise _fail("semantic candidate generation supplements are invalid")
    extended = (
        extend_semantic_candidate_lattice(
            lattice,
            supplemental_groups=supplements,
        )
        if supplements
        else lattice
    )
    validated_output = validate_semantic_candidate_lattice(
        value.get("outputLattice"),
        expected_transcript_sha256=canonical_json_sha256(expected_document),
    )
    generation_body = {
        "inputLatticeSha256": lattice["latticeSha256"],
        "arbitrationDecisionSha256": arbitration["decisionSha256"],
        "supplementalGroups": supplements,
        "outputLatticeSha256": extended["latticeSha256"],
    }
    generation_sha = canonical_json_sha256(generation_body)
    expected_binding = {
        "sourceMediaSha256": lattice["binding"]["sourceMediaSha256"],
        "transcriptSha256": canonical_json_sha256(expected_document),
        "inputLatticeSha256": lattice["latticeSha256"],
        "arbitrationArtifactSha256": canonical_json_sha256(arbitration),
        "arbitrationDecisionSha256": arbitration["decisionSha256"],
        "outputLatticeSha256": extended["latticeSha256"],
    }
    raw_fulfilled = value.get("fulfilledRequests")
    raw_unfulfilled = value.get("unfulfilledRequests")
    if not isinstance(raw_fulfilled, list) or not isinstance(
        raw_unfulfilled,
        list,
    ):
        raise _fail("semantic candidate generation fulfillment trace is invalid")
    supplemented_group_ids = {
        str(item.get("groupId"))
        for item in supplements
        if isinstance(item, Mapping)
    }
    handled_requests = [
        request
        for request in arbitration["candidateGenerationRequests"]
        if str(request["groupId"]) in supplemented_group_ids
    ]
    expected_unfulfilled = [
        dict(request)
        for request in arbitration["candidateGenerationRequests"]
        if str(request["groupId"]) not in supplemented_group_ids
    ]
    expected_fulfilled = _fulfilled_trace(
        handled_requests,
        before_lattice=lattice,
        after_lattice=extended,
        supplements=supplements,
    )
    expected_metrics = {
        "requestCount": len(arbitration["candidateGenerationRequests"]),
        "fulfilledRequestCount": len(expected_fulfilled),
        "unfulfilledRequestCount": len(expected_unfulfilled),
        "generatedCandidateCount": sum(
            len(item.get("generatedCandidateIds", []))
            for item in expected_fulfilled
        ),
        "inputAvailableGroupCount": lattice["availability"][
            "availableGroupCount"
        ],
        "outputAvailableGroupCount": extended["availability"][
            "availableGroupCount"
        ],
    }
    if (
        value.get("artifactId")
        != "semantic-generation-" + generation_sha[:24]
        or value.get("generationSha256") != generation_sha
        or value.get("binding") != expected_binding
        or validated_output != extended
        or raw_fulfilled != expected_fulfilled
        or raw_unfulfilled != expected_unfulfilled
        or value.get("status")
        != ("completed" if not expected_unfulfilled else "partial")
        or value.get("metrics") != expected_metrics
        or len(expected_fulfilled)
        + len(expected_unfulfilled)
        != len(arbitration["candidateGenerationRequests"])
    ):
        raise _fail(
            "semantic candidate generation identity or derived fields are "
            "inconsistent"
        )
    return value


__all__ = [
    "SEMANTIC_CANDIDATE_GENERATION_ARTIFACT_TYPE",
    "SEMANTIC_CANDIDATE_GENERATION_SCHEMA_VERSION",
    "SemanticCandidateGenerationRegistry",
    "SemanticCandidateGenerator",
    "build_asr_text_challenger_result",
    "build_open_set_lid_challenger_result",
    "build_timeline_challenger_result",
    "build_voice_activity_challenger_result",
    "validate_semantic_candidate_generation",
]
