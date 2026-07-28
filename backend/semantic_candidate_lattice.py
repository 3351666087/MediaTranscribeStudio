"""Hash-bound cross-model candidates for mandatory semantic arbitration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .asr_evidence import ASR_CANDIDATE_SET_KEYS, validate_asr_candidate_set
from .language import normalize_language_tag
from .persistence import canonical_json_sha256, validate_strict_json


SEMANTIC_CANDIDATE_LATTICE_SCHEMA_VERSION = "1.0.0"
SEMANTIC_CANDIDATE_LATTICE_ARTIFACT_TYPE = "semantic-candidate-lattice"
SEMANTIC_CANDIDATE_DOMAINS = (
    "speech-disposition",
    "speaker-cardinality-timeline",
    "speaker-assignment",
    "language-span",
    "asr-text",
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SPEAKER_ID = re.compile(r"^speaker-[1-9][0-9]*$")
_IDENTITY_STATUSES = {
    "manifest-bound",
    "artifact-bound",
    "human-attested",
    "unverified",
}
_PRODUCER_TYPES = {"model", "deterministic", "human"}
_TIMELINE_KINDS = {
    "current-transcript",
    "overlap-preserving",
    "single-speaker",
    "challenger",
}
_GROUP_STATUSES = {"available", "candidate-domain-unavailable"}
_DOMAIN_STATUSES = {
    "available",
    "partial",
    "candidate-domain-unavailable",
}


class SemanticCandidateLatticeError(ValueError):
    """Raised when a semantic candidate lattice is malformed or rebound."""


def _fail(message: str) -> SemanticCandidateLatticeError:
    return SemanticCandidateLatticeError(message)


def _text(value: Any, field: str, *, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise _fail(f"{field} exceeds {maximum} characters")
    return result


def _sha256(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise _fail(f"{field} must be <= {maximum}")
    return value


def _optional_score(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise _fail(f"{field} must be finite")
    return result


def _producer(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    required = {
        "producerType",
        "systemId",
        "revision",
        "artifactSha256",
        "modelManifestSha256",
        "identityStatus",
    }
    if set(value) != required:
        raise _fail(f"{field} fields do not match the producer contract")
    producer_type = value.get("producerType")
    identity_status = value.get("identityStatus")
    if producer_type not in _PRODUCER_TYPES:
        raise _fail(f"{field}.producerType is unsupported")
    if identity_status not in _IDENTITY_STATUSES:
        raise _fail(f"{field}.identityStatus is unsupported")
    manifest_sha = _sha256(
        value.get("modelManifestSha256"),
        f"{field}.modelManifestSha256",
        nullable=True,
    )
    if identity_status == "manifest-bound" and manifest_sha is None:
        raise _fail(f"{field} manifest-bound identity requires a model manifest")
    if producer_type == "human" and identity_status != "human-attested":
        raise _fail(f"{field} human producers must be human-attested")
    return {
        "producerType": producer_type,
        "systemId": _text(value.get("systemId"), f"{field}.systemId", maximum=160),
        "revision": _text(value.get("revision"), f"{field}.revision", maximum=160),
        "artifactSha256": _sha256(
            value.get("artifactSha256"),
            f"{field}.artifactSha256",
        ),
        "modelManifestSha256": manifest_sha,
        "identityStatus": identity_status,
    }


def _time_range(
    payload: Mapping[str, Any],
    *,
    field: str,
    duration_ms: int,
) -> tuple[int, int]:
    start = _integer(payload.get("startMs"), f"{field}.startMs")
    end = _integer(payload.get("endMs"), f"{field}.endMs", minimum=1)
    if end <= start or end > duration_ms:
        raise _fail(f"{field} time range is invalid or outside source media")
    return start, end


def _speaker_ids(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise _fail(f"{field} must be a non-empty array")
    result = [str(item) for item in value]
    if len(set(result)) != len(result):
        raise _fail(f"{field} must contain unique speaker IDs")
    if any(_SPEAKER_ID.fullmatch(item) is None for item in result):
        raise _fail(f"{field} contains an invalid canonical speaker ID")
    expected = [
        f"speaker-{index}"
        for index in range(1, len(result) + 1)
    ]
    if result != expected:
        raise _fail(f"{field} must be the contiguous speaker-1..speaker-N set")
    return result


def _normalize_turns(
    value: Any,
    *,
    field: str,
    duration_ms: int,
    allowed_speakers: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise _fail(f"{field} must be a non-empty array")
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise _fail(f"{field}[{index}] must be an object")
        if set(raw) not in (
            {"startMs", "endMs", "speakerId", "overlap"},
            {"startMs", "endMs", "speakerId", "overlap", "text"},
        ):
            raise _fail(f"{field}[{index}] fields do not match the turn contract")
        start, end = _time_range(
            raw,
            field=f"{field}[{index}]",
            duration_ms=duration_ms,
        )
        speaker_id = str(raw.get("speakerId") or "")
        if speaker_id not in allowed_speakers:
            raise _fail(f"{field}[{index}].speakerId is not canonical")
        overlap = raw.get("overlap")
        if not isinstance(overlap, bool):
            raise _fail(f"{field}[{index}].overlap must be boolean")
        turn = {
            "startMs": start,
            "endMs": end,
            "speakerId": speaker_id,
            "overlap": overlap,
        }
        if "text" in raw:
            turn["text"] = _text(
                raw.get("text"),
                f"{field}[{index}].text",
                maximum=20_000,
            )
        turns.append(turn)
    canonical = sorted(
        turns,
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["speakerId"],
            item["overlap"],
        ),
    )
    if turns != canonical:
        raise _fail(f"{field} must use canonical chronological ordering")
    return turns


def _payload(
    domain: str,
    value: Any,
    *,
    duration_ms: int,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object")
    if domain == "speech-disposition":
        required = {"classification", "startMs", "endMs"}
        evidence_fields = {
            "speechDurationMs",
            "speechRatio",
            "speechWindowCount",
        }
        if set(value) not in (required, required | evidence_fields):
            raise _fail(f"{field} fields do not match speech-disposition")
        classification = value.get("classification")
        if classification not in {
            "transcribable-speech",
            "no-transcribable-speech",
        }:
            raise _fail(f"{field}.classification is unsupported")
        start, end = _time_range(value, field=field, duration_ms=duration_ms)
        if start != 0 or end != duration_ms:
            raise _fail(f"{field} must cover the complete source media")
        payload = {
            "classification": classification,
            "startMs": start,
            "endMs": end,
        }
        if evidence_fields.issubset(value):
            speech_duration = _integer(
                value.get("speechDurationMs"),
                f"{field}.speechDurationMs",
            )
            speech_window_count = _integer(
                value.get("speechWindowCount"),
                f"{field}.speechWindowCount",
            )
            speech_ratio = _optional_score(
                value.get("speechRatio"),
                f"{field}.speechRatio",
            )
            if (
                speech_duration > duration_ms
                or speech_ratio is None
                or not 0.0 <= speech_ratio <= 1.0
                or abs(speech_ratio - speech_duration / duration_ms) > 1e-9
                or (speech_duration == 0) != (speech_window_count == 0)
            ):
                raise _fail(f"{field} speech evidence metrics are inconsistent")
            payload.update(
                {
                    "speechDurationMs": speech_duration,
                    "speechRatio": speech_ratio,
                    "speechWindowCount": speech_window_count,
                }
            )
        return payload
    if domain == "speaker-cardinality-timeline":
        required = {
            "speakerCount",
            "speakerIds",
            "timelineKind",
            "startMs",
            "endMs",
            "turns",
        }
        if set(value) != required:
            raise _fail(
                f"{field} fields do not match speaker-cardinality-timeline"
            )
        speaker_ids = _speaker_ids(value.get("speakerIds"), f"{field}.speakerIds")
        count = _integer(value.get("speakerCount"), f"{field}.speakerCount", minimum=1)
        if count != len(speaker_ids):
            raise _fail(f"{field}.speakerCount does not match speakerIds")
        timeline_kind = value.get("timelineKind")
        if timeline_kind not in _TIMELINE_KINDS:
            raise _fail(f"{field}.timelineKind is unsupported")
        start, end = _time_range(value, field=field, duration_ms=duration_ms)
        if start != 0 or end != duration_ms:
            raise _fail(f"{field} must bind a complete-media timeline")
        turns = _normalize_turns(
            value.get("turns"),
            field=f"{field}.turns",
            duration_ms=duration_ms,
            allowed_speakers=set(speaker_ids),
        )
        observed = {turn["speakerId"] for turn in turns}
        if observed != set(speaker_ids):
            raise _fail(f"{field}.turns must observe every declared speaker")
        return {
            "speakerCount": count,
            "speakerIds": speaker_ids,
            "timelineKind": timeline_kind,
            "startMs": start,
            "endMs": end,
            "turns": turns,
        }
    if domain == "speaker-assignment":
        required = {"segmentId", "startMs", "endMs", "speakerId", "score"}
        if set(value) != required:
            raise _fail(f"{field} fields do not match speaker-assignment")
        start, end = _time_range(value, field=field, duration_ms=duration_ms)
        speaker_id = _text(
            value.get("speakerId"),
            f"{field}.speakerId",
            maximum=80,
        )
        if _SPEAKER_ID.fullmatch(speaker_id) is None:
            raise _fail(f"{field}.speakerId is invalid")
        return {
            "segmentId": _text(
                value.get("segmentId"),
                f"{field}.segmentId",
                maximum=160,
            ),
            "startMs": start,
            "endMs": end,
            "speakerId": speaker_id,
            "score": _optional_score(value.get("score"), f"{field}.score"),
        }
    if domain == "language-span":
        required = {
            "segmentId",
            "startMs",
            "endMs",
            "language",
            "confidence",
        }
        if set(value) not in (required, required | {"evidenceSha256"}):
            raise _fail(f"{field} fields do not match language-span")
        start, end = _time_range(value, field=field, duration_ms=duration_ms)
        try:
            language = normalize_language_tag(
                value.get("language"),
                allow_auto=False,
            )
        except ValueError as exc:
            raise _fail(f"{field}.language is invalid") from exc
        confidence = _optional_score(value.get("confidence"), f"{field}.confidence")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise _fail(f"{field}.confidence must be between 0 and 1")
        payload = {
            "segmentId": _text(
                value.get("segmentId"),
                f"{field}.segmentId",
                maximum=160,
            ),
            "startMs": start,
            "endMs": end,
            "language": language,
            "confidence": confidence,
        }
        if "evidenceSha256" in value:
            payload["evidenceSha256"] = _sha256(
                value.get("evidenceSha256"),
                f"{field}.evidenceSha256",
            )
        return payload
    if domain == "asr-text":
        required = {
            "segmentId",
            "startMs",
            "endMs",
            "text",
            "language",
            "sourceCandidateId",
            "candidateSetSha256",
        }
        if set(value) != required:
            raise _fail(f"{field} fields do not match asr-text")
        start, end = _time_range(value, field=field, duration_ms=duration_ms)
        try:
            language = normalize_language_tag(
                value.get("language"),
                allow_auto=False,
            )
        except ValueError as exc:
            raise _fail(f"{field}.language is invalid") from exc
        source_candidate_id = value.get("sourceCandidateId")
        if source_candidate_id is not None:
            source_candidate_id = _text(
                source_candidate_id,
                f"{field}.sourceCandidateId",
                maximum=160,
            )
        return {
            "segmentId": _text(
                value.get("segmentId"),
                f"{field}.segmentId",
                maximum=160,
            ),
            "startMs": start,
            "endMs": end,
            "text": _text(value.get("text"), f"{field}.text", maximum=20_000),
            "language": language,
            "sourceCandidateId": source_candidate_id,
            "candidateSetSha256": _sha256(
                value.get("candidateSetSha256"),
                f"{field}.candidateSetSha256",
                nullable=True,
            ),
        }
    raise _fail(f"unsupported semantic candidate domain: {domain}")


def _candidate(
    raw: Mapping[str, Any],
    *,
    domain: str,
    scope_id: str,
    duration_ms: int,
    field: str,
) -> tuple[dict[str, Any], bool]:
    required = {
        "payload",
        "producers",
        "selectionEligible",
        "eligibilityReason",
        "isCurrent",
    }
    if set(raw) != required:
        raise _fail(f"{field} fields do not match the candidate input contract")
    raw_producers = raw.get("producers")
    if not isinstance(raw_producers, list) or not raw_producers:
        raise _fail(f"{field}.producers must be a non-empty array")
    producers = [
        _producer(item, f"{field}.producers[{index}]")
        for index, item in enumerate(raw_producers)
    ]
    producers = sorted(
        {canonical_json_sha256(item): item for item in producers}.values(),
        key=canonical_json_sha256,
    )
    eligible = raw.get("selectionEligible")
    if not isinstance(eligible, bool):
        raise _fail(f"{field}.selectionEligible must be boolean")
    reason = _text(
        raw.get("eligibilityReason"),
        f"{field}.eligibilityReason",
        maximum=160,
    )
    if eligible and reason != "eligible":
        raise _fail(f"{field} eligible candidates must use reason=eligible")
    if not eligible and reason == "eligible":
        raise _fail(f"{field} ineligible candidates require a failure reason")
    if eligible and not any(
        producer["identityStatus"] != "unverified"
        for producer in producers
    ):
        raise _fail(f"{field} cannot be eligible with only unverified producers")
    is_current = raw.get("isCurrent")
    if not isinstance(is_current, bool):
        raise _fail(f"{field}.isCurrent must be boolean")
    payload = _payload(
        domain,
        raw.get("payload"),
        duration_ms=duration_ms,
        field=f"{field}.payload",
    )
    payload_sha = canonical_json_sha256(payload)
    body = {
        "domain": domain,
        "scopeId": scope_id,
        "payloadSha256": payload_sha,
        "selectionEligible": eligible,
        "eligibilityReason": reason,
        "producers": producers,
        "payload": payload,
    }
    return {
        "candidateId": "candidate-" + canonical_json_sha256(body)[:24],
        **body,
    }, is_current


def _group(
    raw: Mapping[str, Any],
    *,
    domain: str,
    duration_ms: int,
    field: str,
) -> dict[str, Any]:
    if set(raw) != {"scopeId", "candidates"}:
        raise _fail(f"{field} fields do not match the candidate group contract")
    scope_id = _text(raw.get("scopeId"), f"{field}.scopeId", maximum=200)
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise _fail(f"{field}.candidates must be a non-empty array")
    by_payload: dict[str, dict[str, Any]] = {}
    current_payloads: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise _fail(f"{field}.candidates[{index}] must be an object")
        candidate, is_current = _candidate(
            raw_candidate,
            domain=domain,
            scope_id=scope_id,
            duration_ms=duration_ms,
            field=f"{field}.candidates[{index}]",
        )
        payload_sha = str(candidate["payloadSha256"])
        if is_current:
            current_payloads.add(payload_sha)
        previous = by_payload.get(payload_sha)
        if previous is None:
            by_payload[payload_sha] = candidate
            continue
        merged_producers = sorted(
            {
                canonical_json_sha256(item): item
                for item in [
                    *previous["producers"],
                    *candidate["producers"],
                ]
            }.values(),
            key=canonical_json_sha256,
        )
        eligible = bool(
            previous["selectionEligible"] or candidate["selectionEligible"]
        )
        reason = (
            "eligible"
            if eligible
            else (
                previous["eligibilityReason"]
                if previous["eligibilityReason"] == candidate["eligibilityReason"]
                else "no-eligible-producer"
            )
        )
        body = {
            "domain": domain,
            "scopeId": scope_id,
            "payloadSha256": payload_sha,
            "selectionEligible": eligible,
            "eligibilityReason": reason,
            "producers": merged_producers,
            "payload": previous["payload"],
        }
        by_payload[payload_sha] = {
            "candidateId": "candidate-" + canonical_json_sha256(body)[:24],
            **body,
        }
    if len(current_payloads) != 1:
        raise _fail(f"{field} must declare exactly one current payload")
    candidates = sorted(
        by_payload.values(),
        key=lambda item: (
            item["payloadSha256"] not in current_payloads,
            item["candidateId"],
        ),
    )
    current = next(
        item
        for item in candidates
        if item["payloadSha256"] in current_payloads
    )
    eligible_count = sum(item["selectionEligible"] for item in candidates)
    current_eligible = bool(current["selectionEligible"])
    if current_eligible and eligible_count >= 2:
        status = "available"
        unavailable_reason = None
    elif not current_eligible:
        status = "candidate-domain-unavailable"
        unavailable_reason = "current-candidate-ineligible"
    elif eligible_count == 0:
        status = "candidate-domain-unavailable"
        unavailable_reason = "no-eligible-candidate"
    else:
        status = "candidate-domain-unavailable"
        unavailable_reason = "single-eligible-candidate"
    group_seed = {"domain": domain, "scopeId": scope_id}
    return {
        "groupId": "group-" + canonical_json_sha256(group_seed)[:24],
        "scopeId": scope_id,
        "status": status,
        "unavailableReason": unavailable_reason,
        "currentCandidateId": current["candidateId"],
        "candidateCount": len(candidates),
        "eligibleCandidateCount": eligible_count,
        "candidates": candidates,
    }


def build_semantic_candidate_lattice(
    *,
    source_media_sha256: str,
    transcript_sha256: str,
    transcript_schema_version: str,
    source_duration_ms: int,
    candidate_groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build a deterministic lattice and derive every availability field."""

    source_sha = _sha256(source_media_sha256, "sourceMediaSha256")
    transcript_sha = _sha256(transcript_sha256, "transcriptSha256")
    schema_version = _text(
        transcript_schema_version,
        "transcriptSchemaVersion",
        maximum=40,
    )
    duration = _integer(source_duration_ms, "sourceDurationMs", minimum=1)
    if not isinstance(candidate_groups, Mapping):
        raise _fail("candidateGroups must be an object")
    unknown = sorted(set(candidate_groups) - set(SEMANTIC_CANDIDATE_DOMAINS))
    if unknown:
        raise _fail(
            "candidateGroups contains unsupported domains: "
            + ", ".join(unknown)
        )

    domains: list[dict[str, Any]] = []
    total_groups = 0
    available_groups = 0
    for domain in SEMANTIC_CANDIDATE_DOMAINS:
        raw_groups = candidate_groups.get(domain, [])
        if (
            not isinstance(raw_groups, Sequence)
            or isinstance(raw_groups, (str, bytes, bytearray))
        ):
            raise _fail(f"candidateGroups.{domain} must be an array")
        groups = [
            _group(
                raw,
                domain=domain,
                duration_ms=duration,
                field=f"candidateGroups.{domain}[{index}]",
            )
            for index, raw in enumerate(raw_groups)
            if isinstance(raw, Mapping)
        ]
        if len(groups) != len(raw_groups):
            raise _fail(f"candidateGroups.{domain} entries must be objects")
        groups.sort(key=lambda item: (item["scopeId"], item["groupId"]))
        if len({item["scopeId"] for item in groups}) != len(groups):
            raise _fail(f"candidateGroups.{domain} scope IDs must be unique")
        available_count = sum(item["status"] == "available" for item in groups)
        unavailable_count = len(groups) - available_count
        if groups and available_count == len(groups):
            status = "available"
            unavailable_reason = None
        elif available_count:
            status = "partial"
            unavailable_reason = "one-or-more-groups-unavailable"
        elif groups:
            status = "candidate-domain-unavailable"
            unavailable_reason = "no-group-has-an-eligible-alternative"
        else:
            status = "candidate-domain-unavailable"
            unavailable_reason = "missing-domain-candidates"
        domains.append(
            {
                "domain": domain,
                "status": status,
                "unavailableReason": unavailable_reason,
                "groupCount": len(groups),
                "availableGroupCount": available_count,
                "unavailableGroupCount": unavailable_count,
                "groups": groups,
            }
        )
        total_groups += len(groups)
        available_groups += available_count
    unavailable_groups = total_groups - available_groups
    body = {
        "schemaVersion": SEMANTIC_CANDIDATE_LATTICE_SCHEMA_VERSION,
        "artifactType": SEMANTIC_CANDIDATE_LATTICE_ARTIFACT_TYPE,
        "binding": {
            "sourceMediaSha256": source_sha,
            "transcriptSha256": transcript_sha,
            "transcriptSchemaVersion": schema_version,
            "sourceDurationMs": duration,
        },
        "requiredDomains": list(SEMANTIC_CANDIDATE_DOMAINS),
        "availability": {
            "allRequiredDomainsAvailable": all(
                item["status"] == "available" for item in domains
            ),
            "requiredDomainCount": len(SEMANTIC_CANDIDATE_DOMAINS),
            "availableDomainCount": sum(
                item["status"] == "available" for item in domains
            ),
            "partialDomainCount": sum(
                item["status"] == "partial" for item in domains
            ),
            "unavailableDomainCount": sum(
                item["status"] == "candidate-domain-unavailable"
                for item in domains
            ),
            "groupCount": total_groups,
            "availableGroupCount": available_groups,
            "unavailableGroupCount": unavailable_groups,
        },
        "domains": domains,
    }
    lattice_sha = canonical_json_sha256(body)
    value = {
        **body,
        "latticeId": "semantic-lattice-" + lattice_sha[:24],
        "latticeSha256": lattice_sha,
    }
    validate_strict_json(value)
    return value


def _transcript_producer(
    transcript_sha256: str,
    transcript_schema_version: str,
) -> dict[str, Any]:
    return {
        "producerType": "deterministic",
        "systemId": "canonical-transcript",
        "revision": transcript_schema_version,
        "artifactSha256": transcript_sha256,
        "modelManifestSha256": None,
        "identityStatus": "artifact-bound",
    }


def _candidate_input(
    *,
    payload: Mapping[str, Any],
    producer: Mapping[str, Any],
    current: bool,
    eligible: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "payload": dict(payload),
        "producers": [dict(producer)],
        "selectionEligible": eligible,
        "eligibilityReason": reason,
        "isCurrent": current,
    }


def _overlap_flags(turns: Sequence[Mapping[str, Any]]) -> list[bool]:
    flags: list[bool] = []
    for index, turn in enumerate(turns):
        start = int(turn["startMs"])
        end = int(turn["endMs"])
        flags.append(
            any(
                other_index != index
                and int(other["endMs"]) > start
                and int(other["startMs"]) < end
                for other_index, other in enumerate(turns)
            )
        )
    return flags


def _asr_candidate_set(
    segment: Mapping[str, Any],
) -> dict[str, Any] | None:
    evidence = segment.get("evidence")
    asr = evidence.get("asr") if isinstance(evidence, Mapping) else None
    if not isinstance(asr, Mapping) or not set(ASR_CANDIDATE_SET_KEYS).issubset(asr):
        return None
    try:
        return validate_asr_candidate_set(
            asr,
            expected_text=str(segment["rawText"]),
            expected_start_ms=int(segment["startMs"]),
            expected_end_ms=int(segment["endMs"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def _asr_producer(candidate_set: Mapping[str, Any]) -> dict[str, Any]:
    identity_status = str(candidate_set["modelIdentityStatus"])
    return {
        "producerType": "model",
        "systemId": str(candidate_set["modelId"]),
        "revision": str(candidate_set["modelRevision"]),
        "artifactSha256": str(candidate_set["candidateSetSha256"]),
        "modelManifestSha256": str(candidate_set["modelManifestSha256"]),
        "identityStatus": (
            "manifest-bound"
            if identity_status == "manifest-bound"
            else "unverified"
        ),
    }


def _legacy_asr_producer(
    asr: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "producerType": "model",
        "systemId": str(asr.get("provider") or "legacy-asr"),
        "revision": str(asr.get("modelRevision") or "unversioned"),
        "artifactSha256": canonical_json_sha256(asr),
        "modelManifestSha256": None,
        "identityStatus": "unverified",
    }


def build_semantic_candidate_lattice_from_document(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive truthful current/alternative domains from a transcript artifact."""

    try:
        validate_strict_json(dict(document))
    except ValueError as exc:
        raise _fail("transcript document is not strict finite JSON") from exc
    transcript_schema_version = _text(
        document.get("schemaVersion"),
        "document.schemaVersion",
        maximum=40,
    )
    source = document.get("source")
    segments = document.get("segments")
    if (
        not isinstance(source, Mapping)
        or not isinstance(segments, list)
        or not segments
    ):
        raise _fail("candidate lattice requires transcript source and segments")
    transcript_sha = canonical_json_sha256(document)
    source_sha = _sha256(source.get("sha256"), "document.source.sha256")
    duration = _integer(
        source.get("durationMs"),
        "document.source.durationMs",
        minimum=1,
    )
    transcript_producer = _transcript_producer(
        transcript_sha,
        transcript_schema_version,
    )
    speaker_policy = document.get("speakerPolicy")
    raw_speaker_ids = (
        speaker_policy.get("speakerIds")
        if isinstance(speaker_policy, Mapping)
        else None
    )
    speaker_ids = _speaker_ids(raw_speaker_ids, "document.speakerPolicy.speakerIds")

    candidate_groups: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in SEMANTIC_CANDIDATE_DOMAINS
    }
    candidate_groups["speech-disposition"].append(
        {
            "scopeId": "media",
            "candidates": [
                _candidate_input(
                    payload={
                        "classification": "transcribable-speech",
                        "startMs": 0,
                        "endMs": duration,
                    },
                    producer=transcript_producer,
                    current=True,
                    eligible=True,
                    reason="eligible",
                )
            ],
        }
    )

    baseline_turns: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise _fail(f"document.segments[{index}] must be an object")
        baseline_turns.append(
            {
                "startMs": _integer(
                    segment.get("startMs"),
                    f"document.segments[{index}].startMs",
                ),
                "endMs": _integer(
                    segment.get("endMs"),
                    f"document.segments[{index}].endMs",
                    minimum=1,
                ),
                "speakerId": str(segment.get("speakerId") or ""),
                "overlap": bool(segment.get("overlapping", False)),
            }
        )
    baseline_turns.sort(
        key=lambda item: (item["startMs"], item["endMs"], item["speakerId"])
    )
    timeline_candidates = [
        _candidate_input(
            payload={
                "speakerCount": len(speaker_ids),
                "speakerIds": speaker_ids,
                "timelineKind": "current-transcript",
                "startMs": 0,
                "endMs": duration,
                "turns": baseline_turns,
            },
            producer=transcript_producer,
            current=True,
            eligible=True,
            reason="eligible",
        )
    ]
    speaker_timeline = document.get("speakerTimeline")
    if isinstance(speaker_timeline, Mapping):
        provider = speaker_timeline.get("provider")
        for name, kind in (
            ("regular", "overlap-preserving"),
            ("exclusive", "single-speaker"),
        ):
            raw_timeline = speaker_timeline.get(name)
            turns = (
                raw_timeline.get("turns")
                if isinstance(raw_timeline, Mapping)
                else None
            )
            digest = (
                raw_timeline.get("sha256")
                if isinstance(raw_timeline, Mapping)
                else None
            )
            if not isinstance(turns, list) or not turns or _SHA256.fullmatch(
                str(digest or "")
            ) is None:
                continue
            flags = _overlap_flags(turns)
            canonical_turns = [
                {
                    "startMs": int(turn["startMs"]),
                    "endMs": int(turn["endMs"]),
                    "speakerId": str(turn["speakerId"]),
                    "overlap": flags[index],
                }
                for index, turn in enumerate(turns)
                if isinstance(turn, Mapping)
            ]
            if len(canonical_turns) != len(turns):
                continue
            native_speakers = sorted(
                {turn["speakerId"] for turn in canonical_turns},
                key=lambda item: int(item.removeprefix("speaker-")),
            )
            if native_speakers != speaker_ids:
                continue
            native_producer = {
                "producerType": "model",
                "systemId": str(
                    provider.get("id")
                    if isinstance(provider, Mapping)
                    else "speaker-timeline"
                ),
                "revision": str(
                    provider.get("version")
                    if isinstance(provider, Mapping)
                    else "unversioned"
                ),
                "artifactSha256": str(digest),
                "modelManifestSha256": None,
                "identityStatus": "artifact-bound",
            }
            timeline_candidates.append(
                _candidate_input(
                    payload={
                        "speakerCount": len(speaker_ids),
                        "speakerIds": speaker_ids,
                        "timelineKind": kind,
                        "startMs": 0,
                        "endMs": duration,
                        "turns": canonical_turns,
                    },
                    producer=native_producer,
                    current=False,
                    eligible=True,
                    reason="eligible",
                )
            )
    candidate_groups["speaker-cardinality-timeline"].append(
        {"scopeId": "media", "candidates": timeline_candidates}
    )

    for index, segment in enumerate(segments):
        assert isinstance(segment, Mapping)
        segment_id = _text(
            segment.get("id"),
            f"document.segments[{index}].id",
            maximum=160,
        )
        start = int(segment["startMs"])
        end = int(segment["endMs"])
        current_speaker = str(segment.get("speakerId") or "")
        raw_scores = segment.get("speakerScores")
        scores = (
            [
                item
                for item in raw_scores
                if isinstance(item, Mapping)
                and _SPEAKER_ID.fullmatch(str(item.get("speakerId") or ""))
                is not None
            ]
            if isinstance(raw_scores, list)
            else []
        )
        score_by_speaker = {
            str(item["speakerId"]): _optional_score(
                item.get("score"),
                f"document.segments[{index}].speakerScores.score",
            )
            for item in scores
        }
        speaker_evidence_sha = canonical_json_sha256(scores)
        speaker_producer = {
            "producerType": "deterministic",
            "systemId": "canonical-speaker-score-evidence",
            "revision": "speaker-score-v1",
            "artifactSha256": speaker_evidence_sha,
            "modelManifestSha256": None,
            "identityStatus": "artifact-bound",
        }
        assignment_candidates: list[dict[str, Any]] = []
        assignment_speakers = [
            current_speaker,
            *sorted(
                (
                    speaker_id
                    for speaker_id in score_by_speaker
                    if speaker_id != current_speaker
                ),
                key=lambda item: int(item.removeprefix("speaker-")),
            ),
        ]
        for speaker_id in assignment_speakers:
            assignment_candidates.append(
                _candidate_input(
                    payload={
                        "segmentId": segment_id,
                        "startMs": start,
                        "endMs": end,
                        "speakerId": speaker_id,
                        "score": score_by_speaker.get(speaker_id),
                    },
                    producer=speaker_producer,
                    current=speaker_id == current_speaker,
                    eligible=True,
                    reason="eligible",
                )
            )
        candidate_groups["speaker-assignment"].append(
            {
                "scopeId": f"segment:{segment_id}",
                "candidates": assignment_candidates,
            }
        )

        evidence = segment.get("evidence")
        asr = evidence.get("asr") if isinstance(evidence, Mapping) else None
        asr = asr if isinstance(asr, Mapping) else {}
        validated_asr = _asr_candidate_set(segment)
        segment_language = str(
            segment.get("language")
            or document.get("language")
            or "und"
        )
        language_candidates = [
            _candidate_input(
                payload={
                    "segmentId": segment_id,
                    "startMs": start,
                    "endMs": end,
                    "language": segment_language,
                    "confidence": None,
                },
                producer=transcript_producer,
                current=True,
                eligible=True,
                reason="eligible",
            )
        ]
        raw_languages = asr.get("languageCandidates")
        if isinstance(raw_languages, str):
            raw_languages = [raw_languages]
        if isinstance(raw_languages, Sequence) and not isinstance(
            raw_languages,
            (str, bytes, bytearray),
        ):
            language_producer = (
                _asr_producer(validated_asr)
                if validated_asr is not None
                else _legacy_asr_producer(asr)
            )
            language_eligible = language_producer["identityStatus"] != "unverified"
            for raw_language in raw_languages:
                try:
                    language = normalize_language_tag(
                        raw_language,
                        allow_auto=False,
                    )
                except ValueError:
                    continue
                if language == segment_language:
                    continue
                language_candidates.append(
                    _candidate_input(
                        payload={
                            "segmentId": segment_id,
                            "startMs": start,
                            "endMs": end,
                            "language": language,
                            "confidence": None,
                        },
                        producer=language_producer,
                        current=False,
                        eligible=language_eligible,
                        reason=(
                            "eligible"
                            if language_eligible
                            else "producer-identity-unverified"
                        ),
                    )
                )
        candidate_groups["language-span"].append(
            {
                "scopeId": f"segment:{segment_id}",
                "candidates": language_candidates,
            }
        )

        if validated_asr is not None:
            asr_producer = _asr_producer(validated_asr)
            text_candidates = [
                _candidate_input(
                    payload={
                        "segmentId": segment_id,
                        "startMs": start,
                        "endMs": end,
                        "text": str(item["text"]),
                        "language": str(item["language"]),
                        "sourceCandidateId": str(item["candidateId"]),
                        "candidateSetSha256": str(
                            validated_asr["candidateSetSha256"]
                        ),
                    },
                    producer=asr_producer,
                    current=rank == 0,
                    eligible=(
                        rank == 0
                        or (
                            item["lexicalRepairEligible"] is True
                            and asr_producer["identityStatus"] != "unverified"
                        )
                    ),
                    reason=(
                        "eligible"
                        if (
                            rank == 0
                            or (
                                item["lexicalRepairEligible"] is True
                                and asr_producer["identityStatus"] != "unverified"
                            )
                        )
                        else "asr-candidate-ineligible"
                    ),
                )
                for rank, item in enumerate(validated_asr["nBest"])
            ]
        else:
            text_candidates = [
                _candidate_input(
                    payload={
                        "segmentId": segment_id,
                        "startMs": start,
                        "endMs": end,
                        "text": str(segment["rawText"]),
                        "language": segment_language,
                        "sourceCandidateId": None,
                        "candidateSetSha256": None,
                    },
                    producer=transcript_producer,
                    current=True,
                    eligible=True,
                    reason="eligible",
                )
            ]
            raw_nbest = asr.get("nBest")
            if isinstance(raw_nbest, list):
                legacy_producer = _legacy_asr_producer(asr)
                for raw_candidate in raw_nbest:
                    if not isinstance(raw_candidate, Mapping):
                        continue
                    candidate_text = raw_candidate.get("text")
                    if (
                        not isinstance(candidate_text, str)
                        or not candidate_text.strip()
                        or candidate_text.strip() == str(segment["rawText"])
                    ):
                        continue
                    try:
                        candidate_language = normalize_language_tag(
                            raw_candidate.get("language") or segment_language,
                            allow_auto=False,
                        )
                    except ValueError:
                        continue
                    text_candidates.append(
                        _candidate_input(
                            payload={
                                "segmentId": segment_id,
                                "startMs": start,
                                "endMs": end,
                                "text": candidate_text.strip(),
                                "language": candidate_language,
                                "sourceCandidateId": (
                                    str(raw_candidate["candidateId"])
                                    if raw_candidate.get("candidateId") is not None
                                    else None
                                ),
                                "candidateSetSha256": None,
                            },
                            producer=legacy_producer,
                            current=False,
                            eligible=False,
                            reason="candidate-set-unbound",
                        )
                    )
        candidate_groups["asr-text"].append(
            {
                "scopeId": f"segment:{segment_id}",
                "candidates": text_candidates,
            }
        )

    return build_semantic_candidate_lattice(
        source_media_sha256=str(source_sha),
        transcript_sha256=transcript_sha,
        transcript_schema_version=transcript_schema_version,
        source_duration_ms=duration,
        candidate_groups=candidate_groups,
    )


def validate_semantic_candidate_lattice(
    lattice: Mapping[str, Any],
    *,
    expected_source_media_sha256: str | None = None,
    expected_transcript_sha256: str | None = None,
) -> dict[str, Any]:
    """Rebuild a persisted lattice and reject any structural or hash tampering."""

    if not isinstance(lattice, Mapping):
        raise _fail("semantic candidate lattice must be an object")
    value = dict(lattice)
    try:
        validate_strict_json(value)
    except ValueError as exc:
        raise _fail(
            "semantic candidate lattice must contain strict finite JSON"
        ) from exc
    required = {
        "schemaVersion",
        "artifactType",
        "latticeId",
        "latticeSha256",
        "binding",
        "requiredDomains",
        "availability",
        "domains",
    }
    if set(value) != required:
        raise _fail("semantic candidate lattice fields do not match schema 1.0.0")
    if (
        value.get("schemaVersion") != SEMANTIC_CANDIDATE_LATTICE_SCHEMA_VERSION
        or value.get("artifactType") != SEMANTIC_CANDIDATE_LATTICE_ARTIFACT_TYPE
    ):
        raise _fail("semantic candidate lattice identity is invalid")
    binding = value.get("binding")
    domains = value.get("domains")
    if not isinstance(binding, Mapping) or not isinstance(domains, list):
        raise _fail("semantic candidate lattice binding or domains are invalid")
    groups: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in SEMANTIC_CANDIDATE_DOMAINS
    }
    for domain_index, raw_domain in enumerate(domains):
        if not isinstance(raw_domain, Mapping):
            raise _fail(f"domains[{domain_index}] must be an object")
        domain = raw_domain.get("domain")
        raw_groups = raw_domain.get("groups")
        if domain not in groups or not isinstance(raw_groups, list):
            raise _fail(f"domains[{domain_index}] is invalid")
        for group_index, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, Mapping):
                raise _fail(
                    f"domains[{domain_index}].groups[{group_index}] must be an object"
                )
            candidates = raw_group.get("candidates")
            if not isinstance(candidates, list):
                raise _fail(
                    f"domains[{domain_index}].groups[{group_index}].candidates "
                    "must be an array"
                )
            groups[str(domain)].append(
                {
                    "scopeId": raw_group.get("scopeId"),
                    "candidates": [
                        {
                            "payload": candidate.get("payload"),
                            "producers": candidate.get("producers"),
                            "selectionEligible": candidate.get(
                                "selectionEligible"
                            ),
                            "eligibilityReason": candidate.get(
                                "eligibilityReason"
                            ),
                            "isCurrent": (
                                candidate.get("candidateId")
                                == raw_group.get("currentCandidateId")
                            ),
                        }
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                    ],
                }
            )
            if len(groups[str(domain)][-1]["candidates"]) != len(candidates):
                raise _fail(
                    f"domains[{domain_index}].groups[{group_index}] contains "
                    "a non-object candidate"
                )
    rebuilt = build_semantic_candidate_lattice(
        source_media_sha256=binding.get("sourceMediaSha256"),
        transcript_sha256=binding.get("transcriptSha256"),
        transcript_schema_version=binding.get("transcriptSchemaVersion"),
        source_duration_ms=binding.get("sourceDurationMs"),
        candidate_groups=groups,
    )
    if value != rebuilt:
        raise _fail(
            "semantic candidate lattice identity or derived fields are "
            "inconsistent"
        )
    if (
        expected_source_media_sha256 is not None
        and rebuilt["binding"]["sourceMediaSha256"]
        != expected_source_media_sha256
    ):
        raise _fail("semantic candidate lattice is rebound to different source media")
    if (
        expected_transcript_sha256 is not None
        and rebuilt["binding"]["transcriptSha256"]
        != expected_transcript_sha256
    ):
        raise _fail("semantic candidate lattice is rebound to a different transcript")
    return rebuilt


def extend_semantic_candidate_lattice(
    lattice: Mapping[str, Any],
    *,
    supplemental_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Append bound challenger candidates without changing existing identities."""

    validated = validate_semantic_candidate_lattice(lattice)
    if (
        not isinstance(supplemental_groups, Sequence)
        or isinstance(supplemental_groups, (str, bytes, bytearray))
        or not supplemental_groups
    ):
        raise _fail("supplementalGroups must be a non-empty array")
    if len(supplemental_groups) > 128:
        raise _fail("supplementalGroups exceeds the bounded maximum of 128")

    group_index = {
        str(group["groupId"]): {
            "domain": str(domain["domain"]),
            "scopeId": str(group["scopeId"]),
            "group": group,
        }
        for domain in validated["domains"]
        for group in domain["groups"]
    }
    candidate_groups: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in SEMANTIC_CANDIDATE_DOMAINS
    }
    mutable_groups: dict[str, dict[str, Any]] = {}
    for domain in validated["domains"]:
        domain_name = str(domain["domain"])
        for group in domain["groups"]:
            rebuilt_input = {
                "scopeId": group["scopeId"],
                "candidates": [
                    {
                        "payload": candidate["payload"],
                        "producers": candidate["producers"],
                        "selectionEligible": candidate["selectionEligible"],
                        "eligibilityReason": candidate["eligibilityReason"],
                        "isCurrent": (
                            candidate["candidateId"]
                            == group["currentCandidateId"]
                        ),
                    }
                    for candidate in group["candidates"]
                ],
            }
            candidate_groups[domain_name].append(rebuilt_input)
            mutable_groups[str(group["groupId"])] = rebuilt_input

    seen_groups: set[str] = set()
    supplemental_candidate_count = 0
    for index, raw in enumerate(supplemental_groups):
        field = f"supplementalGroups[{index}]"
        if not isinstance(raw, Mapping):
            raise _fail(f"{field} must be an object")
        if set(raw) != {"domain", "groupId", "scopeId", "candidates"}:
            raise _fail(f"{field} fields do not match the supplement contract")
        group_id = _text(raw.get("groupId"), f"{field}.groupId", maximum=200)
        existing = group_index.get(group_id)
        if existing is None:
            raise _fail(f"{field}.groupId is unknown")
        if group_id in seen_groups:
            raise _fail(f"{field}.groupId is duplicated")
        seen_groups.add(group_id)
        if (
            raw.get("domain") != existing["domain"]
            or raw.get("scopeId") != existing["scopeId"]
        ):
            raise _fail(f"{field} is rebound to another domain or scope")
        raw_candidates = raw.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _fail(f"{field}.candidates must be a non-empty array")
        if len(raw_candidates) > 8:
            raise _fail(f"{field}.candidates exceeds the bounded maximum of 8")
        supplemental_candidate_count += len(raw_candidates)
        if supplemental_candidate_count > 256:
            raise _fail("supplemental candidates exceed the bounded maximum of 256")
        for candidate_index, candidate in enumerate(raw_candidates):
            candidate_field = f"{field}.candidates[{candidate_index}]"
            if not isinstance(candidate, Mapping):
                raise _fail(f"{candidate_field} must be an object")
            if set(candidate) != {
                "payload",
                "producers",
                "selectionEligible",
                "eligibilityReason",
            }:
                raise _fail(
                    f"{candidate_field} fields do not match the challenger "
                    "candidate contract"
                )
            try:
                validate_strict_json(dict(candidate))
            except ValueError as exc:
                raise _fail(
                    f"{candidate_field} must contain strict finite JSON"
                ) from exc
            mutable_groups[group_id]["candidates"].append(
                {
                    **dict(candidate),
                    "isCurrent": False,
                }
            )

    binding = validated["binding"]
    extended = build_semantic_candidate_lattice(
        source_media_sha256=binding["sourceMediaSha256"],
        transcript_sha256=binding["transcriptSha256"],
        transcript_schema_version=binding["transcriptSchemaVersion"],
        source_duration_ms=binding["sourceDurationMs"],
        candidate_groups=candidate_groups,
    )
    extended_groups = {
        str(group["groupId"]): group
        for domain in extended["domains"]
        for group in domain["groups"]
    }
    for group_id, previous in group_index.items():
        current = extended_groups.get(group_id)
        if current is None:
            raise _fail("candidate lattice extension removed an existing group")
        if current["currentCandidateId"] != previous["group"]["currentCandidateId"]:
            raise _fail(
                "candidate lattice extension changed the current candidate identity"
            )
        previous_ids = {
            candidate["candidateId"]
            for candidate in previous["group"]["candidates"]
        }
        current_ids = {
            candidate["candidateId"] for candidate in current["candidates"]
        }
        if not previous_ids.issubset(current_ids):
            raise _fail(
                "candidate lattice extension replaced an existing candidate identity"
            )
    if extended["latticeSha256"] == validated["latticeSha256"]:
        raise _fail("candidate lattice extension did not add a distinct candidate")
    return extended


def compact_candidate_lattice_context(
    lattice: Mapping[str, Any],
    *,
    segment_ids: Sequence[str],
) -> dict[str, Any]:
    """Return bounded candidate identities and payloads for one semantic batch."""

    validated = validate_semantic_candidate_lattice(lattice)
    scopes = {"media", *(f"segment:{segment_id}" for segment_id in segment_ids)}

    def summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
        domain = str(candidate["domain"])
        payload = candidate["payload"]
        if domain == "speech-disposition":
            return {"classification": payload["classification"]}
        if domain == "speaker-cardinality-timeline":
            return {
                "speakerCount": payload["speakerCount"],
                "timelineKind": payload["timelineKind"],
                "turnCount": len(payload["turns"]),
            }
        if domain == "speaker-assignment":
            return {
                "speakerId": payload["speakerId"],
                "score": payload["score"],
            }
        if domain == "language-span":
            return {
                "language": payload["language"],
                "confidence": payload["confidence"],
            }
        return {
            "sourceCandidateId": payload["sourceCandidateId"],
            "text": payload["text"],
            "language": payload["language"],
        }

    groups = [
        {
            "domain": domain["domain"],
            "groupId": group["groupId"],
            "scopeId": group["scopeId"],
            "status": group["status"],
            "unavailableReason": group["unavailableReason"],
            "currentCandidateId": group["currentCandidateId"],
            "alternatives": [
                {
                    "candidateId": candidate["candidateId"],
                    "summary": summary(candidate),
                }
                for candidate in group["candidates"]
                if candidate["selectionEligible"]
                and candidate["candidateId"] != group["currentCandidateId"]
            ],
        }
        for domain in validated["domains"]
        for group in domain["groups"]
        if group["scopeId"] in scopes
    ]
    return {
        "latticeId": validated["latticeId"],
        "latticeSha256": validated["latticeSha256"],
        "availability": {
            "allRequiredDomainsAvailable": validated["availability"][
                "allRequiredDomainsAvailable"
            ],
            "unavailableDomains": [
                domain["domain"]
                for domain in validated["domains"]
                if domain["status"] != "available"
            ],
        },
        "domains": [
            {
                "domain": domain["domain"],
                "status": domain["status"],
            }
            for domain in validated["domains"]
        ],
        "groups": groups,
    }


__all__ = [
    "SEMANTIC_CANDIDATE_DOMAINS",
    "SEMANTIC_CANDIDATE_LATTICE_ARTIFACT_TYPE",
    "SEMANTIC_CANDIDATE_LATTICE_SCHEMA_VERSION",
    "SemanticCandidateLatticeError",
    "build_semantic_candidate_lattice",
    "build_semantic_candidate_lattice_from_document",
    "compact_candidate_lattice_context",
    "extend_semantic_candidate_lattice",
    "validate_semantic_candidate_lattice",
]
