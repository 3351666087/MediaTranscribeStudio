from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from backend import (
    MappingLocalLLMProvider,
    SemanticCandidateLatticeError,
    SemanticProcessingRunner,
    WorkerError,
    build_semantic_candidate_lattice,
    build_semantic_candidate_lattice_from_document,
    compact_candidate_lattice_context,
    validate_semantic_candidate_lattice,
    validate_semantic_suggestions_artifact,
)
from backend.asr_evidence import build_asr_candidate_set


ROOT = Path(__file__).resolve().parents[1]


def _asr_nbest() -> dict:
    return build_asr_candidate_set(
        model_id="fixture-asr",
        model_revision="revision-1",
        model_manifest_sha256="b" * 64,
        model_identity_status="manifest-bound",
        source_audio_sha256="a" * 64,
        normalization_profile="mono-16khz-f32-v1",
        source_window_id="segment-1",
        start_ms=0,
        end_ms=1_000,
        hypotheses=[
            {
                "text": "Hello world",
                "language": "en",
                "tokens": [
                    {"text": "Hello", "startMs": 0, "endMs": 400},
                    {"text": "world", "startMs": 500, "endMs": 900},
                ],
                "acousticScore": -0.1,
                "acousticScoreStatus": "available",
                "decodeScore": -0.1,
                "decodeScoreStatus": "available",
            },
            {
                "text": "Hallo world",
                "language": "en",
                "tokens": [
                    {"text": "Hallo", "startMs": 0, "endMs": 400},
                    {"text": "world", "startMs": 500, "endMs": 900},
                ],
                "acousticScore": -0.2,
                "acousticScoreStatus": "available",
                "decodeScore": -0.2,
                "decodeScoreStatus": "available",
            },
        ],
    )


def _segment(
    segment_id: str,
    *,
    start_ms: int,
    speaker_id: str,
    text: str,
    asr: dict,
) -> dict:
    return {
        "id": segment_id,
        "startMs": start_ms,
        "endMs": start_ms + 1_000,
        "speakerId": speaker_id,
        "rawText": text,
        "normalizedText": text,
        "displayText": text,
        "confidence": 0.8,
        "speakerScores": [
            {"speakerId": "speaker-1", "score": 0.55},
            {"speakerId": "speaker-2", "score": 0.45},
        ],
        "speakerMargin": 0.1,
        "overlapping": False,
        "humanLocked": False,
        "revisions": [],
        "language": "en",
        "evidence": {"asr": asr},
    }


def _document(*, legacy_asr: bool = False) -> dict:
    first_asr = (
        {
            "provider": "legacy-fixture",
            "nBest": [
                {
                    "candidateId": "legacy-2",
                    "text": "Hallo world",
                    "language": "en",
                    "lexicalRepairEligible": True,
                }
            ],
        }
        if legacy_asr
        else _asr_nbest()
    )
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-lattice-fixture",
        "jobId": "lattice-fixture",
        "generatedAt": "2026-07-28T00:00:00Z",
        "language": "en",
        "source": {
            "fileName": "fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 2_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": 2,
            "speakerIds": ["speaker-1", "speaker-2"],
        },
        "speakers": [{"id": "speaker-1"}, {"id": "speaker-2"}],
        "segments": [
            _segment(
                "segment-1",
                start_ms=0,
                speaker_id="speaker-1",
                text="Hello world",
                asr=first_asr,
            ),
            _segment(
                "segment-2",
                start_ms=1_000,
                speaker_id="speaker-2",
                text="Acknowledged",
                asr={"provider": "fixture-asr"},
            ),
        ],
        "provenance": {"offline": True, "models": []},
    }


def _domain(lattice: dict, name: str) -> dict:
    return next(item for item in lattice["domains"] if item["domain"] == name)


def test_document_lattice_is_stable_schema_valid_and_truthfully_partial() -> None:
    document = _document()

    first = build_semantic_candidate_lattice_from_document(document)
    second = build_semantic_candidate_lattice_from_document(document)

    assert first == second
    assert validate_semantic_candidate_lattice(first) == first
    assert first["latticeId"] == (
        "semantic-lattice-" + first["latticeSha256"][:24]
    )
    assert first["availability"]["allRequiredDomainsAvailable"] is False
    assert _domain(first, "speech-disposition")["status"] == (
        "candidate-domain-unavailable"
    )
    assert _domain(first, "speaker-assignment")["status"] == "available"
    assert _domain(first, "asr-text")["status"] == "partial"

    schema = json.loads(
        (
            ROOT / "contracts" / "semantic-candidate-lattice.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first)


def test_missing_and_single_candidate_domains_report_unavailable() -> None:
    missing = build_semantic_candidate_lattice(
        source_media_sha256="a" * 64,
        transcript_sha256="b" * 64,
        transcript_schema_version="2.0.0",
        source_duration_ms=1_000,
        candidate_groups={},
    )

    assert missing["availability"] == {
        "allRequiredDomainsAvailable": False,
        "requiredDomainCount": 5,
        "availableDomainCount": 0,
        "partialDomainCount": 0,
        "unavailableDomainCount": 5,
        "groupCount": 0,
        "availableGroupCount": 0,
        "unavailableGroupCount": 0,
    }
    assert all(
        domain["unavailableReason"] == "missing-domain-candidates"
        for domain in missing["domains"]
    )

    derived = build_semantic_candidate_lattice_from_document(_document())
    speech = _domain(derived, "speech-disposition")["groups"][0]
    assert speech["candidateCount"] == 1
    assert speech["eligibleCandidateCount"] == 1
    assert speech["status"] == "candidate-domain-unavailable"
    assert speech["unavailableReason"] == "single-eligible-candidate"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("latticeSha256", "f" * 64),
        lambda value: value["binding"].__setitem__(
            "sourceMediaSha256",
            "f" * 64,
        ),
        lambda value: value["domains"][0].__setitem__("status", "available"),
        lambda value: value["domains"][0]["groups"][0].__setitem__(
            "currentCandidateId",
            "candidate-" + "f" * 24,
        ),
        lambda value: value["domains"][0]["groups"][0]["candidates"][0][
            "payload"
        ].__setitem__("classification", "no-transcribable-speech"),
        lambda value: value["domains"][0]["groups"][0]["candidates"][0][
            "producers"
        ][0].__setitem__("revision", "tampered"),
    ],
)
def test_lattice_rejects_binding_identity_payload_and_derived_field_tampering(
    mutate,
) -> None:
    value = copy.deepcopy(
        build_semantic_candidate_lattice_from_document(_document())
    )
    mutate(value)

    with pytest.raises(SemanticCandidateLatticeError):
        validate_semantic_candidate_lattice(value)


def test_unbound_legacy_nbest_is_visible_but_cannot_authorize_repair() -> None:
    lattice = build_semantic_candidate_lattice_from_document(
        _document(legacy_asr=True)
    )
    group = _domain(lattice, "asr-text")["groups"][0]

    assert group["candidateCount"] == 2
    assert group["eligibleCandidateCount"] == 1
    assert group["status"] == "candidate-domain-unavailable"
    alternative = next(
        item
        for item in group["candidates"]
        if item["candidateId"] != group["currentCandidateId"]
    )
    assert alternative["selectionEligible"] is False
    assert alternative["eligibilityReason"] == "candidate-set-unbound"


def test_compact_context_exposes_choices_without_repeating_full_timelines() -> None:
    lattice = build_semantic_candidate_lattice_from_document(_document())

    context = compact_candidate_lattice_context(
        lattice,
        segment_ids=["segment-1"],
    )

    assert context["latticeSha256"] == lattice["latticeSha256"]
    assert {group["scopeId"] for group in context["groups"]} == {
        "media",
        "segment:segment-1",
    }
    assert all(
        "turns" not in alternative["summary"]
        for group in context["groups"]
        for alternative in group["alternatives"]
    )
    asr_group = next(
        group
        for group in context["groups"]
        if group["domain"] == "asr-text"
    )
    assert asr_group["alternatives"][0]["summary"]["text"] == "Hallo world"


def test_semantic_artifact_binds_lattice_and_legacy_v8_remains_readable() -> None:
    document = _document()
    artifact = SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [
                {
                    "results": [
                        {
                            "segmentId": "segment-1",
                            "decision": "abstain",
                            "confidence": 0.8,
                        },
                        {
                            "segmentId": "segment-2",
                            "decision": "abstain",
                            "confidence": 0.8,
                        },
                    ]
                }
            ]
        ),
        model="fixture",
    ).run(document)

    assert artifact["schemaVersion"] == "1.1.0"
    assert artifact["promptVersion"] == "semantic-candidate-lattice-v9"
    assert (
        artifact["input"]["candidateLatticeSha256"]
        == artifact["candidateLattice"]["latticeSha256"]
    )

    tampered = copy.deepcopy(artifact)
    tampered["candidateLattice"]["domains"][0]["groups"][0]["candidates"][0][
        "payload"
    ]["classification"] = "no-transcribable-speech"
    with pytest.raises(WorkerError, match="candidate lattice"):
        validate_semantic_suggestions_artifact(
            tampered,
            expected_job_id="lattice-fixture",
            expected_transcript_sha256=artifact["input"]["transcriptSha256"],
        )

    legacy = copy.deepcopy(artifact)
    legacy["schemaVersion"] = "1.0.0"
    legacy["promptVersion"] = "semantic-candidate-state-v8"
    legacy["input"].pop("candidateLatticeSha256")
    legacy.pop("candidateLattice")
    assert (
        validate_semantic_suggestions_artifact(
            legacy,
            expected_job_id="lattice-fixture",
            expected_transcript_sha256=artifact["input"]["transcriptSha256"],
        )
        == legacy
    )


def test_provided_lattice_cannot_be_rebound_to_another_transcript() -> None:
    document = _document()
    lattice = build_semantic_candidate_lattice_from_document(document)
    rebound = copy.deepcopy(document)
    rebound["segments"][0]["normalizedText"] = "Changed"

    with pytest.raises(SemanticCandidateLatticeError, match="transcript"):
        SemanticProcessingRunner(
            provider=MappingLocalLLMProvider([]),
            model="fixture",
        ).run(rebound, candidate_lattice=lattice)
