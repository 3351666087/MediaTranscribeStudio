from __future__ import annotations

import copy
import json

import pytest

from backend import (
    LocalLLMConfig,
    LocalLLMContextWindowError,
    MappingLocalLLMProvider,
    SemanticProcessingRunner,
    WorkerError,
    attach_semantic_suggestions_to_review,
    validate_semantic_suggestions_artifact,
)
from backend.persistence import canonical_json_sha256
from backend.semantic_processing import (
    _segment_request,
    _semantic_batch_response_schema,
)


def _segment(
    segment_id: str,
    *,
    speaker_id: str,
    text: str,
    start_ms: int,
    human_locked: bool = False,
    asr: dict | None = None,
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
            {"speakerId": "speaker-1", "score": 0.51},
            {"speakerId": "speaker-2", "score": 0.49},
        ],
        "speakerMargin": 0.02,
        "overlapping": False,
        "humanLocked": human_locked,
        "revisions": [],
        "language": "en",
        "evidence": {"asr": asr or {"provider": "fixture-asr"}},
    }


def _document(*, first_asr: dict | None = None) -> dict:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-semantic-fixture",
        "jobId": "semantic-fixture",
        "generatedAt": "2026-07-26T00:00:00Z",
        "language": "en",
        "source": {
            "fileName": "fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 3_000,
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
                speaker_id="speaker-1",
                text="I can go",
                start_ms=0,
                asr=first_asr,
            ),
            _segment(
                "segment-2",
                speaker_id="speaker-1",
                text="Hello world",
                start_ms=1_000,
            ),
            _segment(
                "segment-3",
                speaker_id="speaker-2",
                text="Acknowledged",
                start_ms=2_000,
            ),
        ],
        "provenance": {"offline": True, "models": []},
    }


def _result(
    segment_id: str,
    *,
    ranking: list[str],
    text: str,
    evidence_refs: list[str],
    candidate_id: str = "",
) -> dict:
    return {
        "segmentId": segment_id,
        "speakerRanking": ranking,
        "normalizedText": text,
        "textEvidenceCandidateId": candidate_id,
        "confidence": 0.8,
        "reasonCodes": ["CONTEXTUAL_CANDIDATE_RERANK"],
        "evidenceRefs": evidence_refs,
    }


def _keep(segment_id: str, speaker_id: str, text: str) -> dict:
    other = "speaker-2" if speaker_id == "speaker-1" else "speaker-1"
    return _result(
        segment_id,
        ranking=[speaker_id, other],
        text=text,
        evidence_refs=[f"segment:{segment_id}"],
    )


def _empty_queue() -> dict:
    return {
        "schemaVersion": "2.0.0",
        "jobId": "semantic-fixture",
        "speakerCountMode": "manual",
        "speakerCountEstimate": {
            "estimatedCount": 2,
            "confidence": 1.0,
            "candidateRange": {"min": 2, "max": 2},
            "method": "manual-hard-constraint",
        },
        "createdAt": "2026-07-26T00:00:00Z",
        "updatedAt": "2026-07-26T00:00:00Z",
        "items": [],
    }


def test_response_schema_binds_each_segment_to_its_evidence_domain() -> None:
    document = _document()
    first = document["segments"][0]
    schema = _semantic_batch_response_schema(
        [
            {
                "segmentId": first["id"],
                "speakerCandidates": first["speakerScores"],
                "asrNBest": [
                    {
                        "candidateId": "nbest-2",
                        "text": "I cannot go",
                        "lexicalRepairEligible": True,
                    }
                ],
                "allowedEvidenceRefs": [
                    "segment:segment-1",
                    "speaker-score:segment-1:speaker-1",
                    "speaker-score:segment-1:speaker-2",
                    "asr-nbest:segment-1:nbest-2",
                ],
            }
        ]
    )

    results = schema["properties"]["results"]
    assert results["minItems"] == results["maxItems"] == 1
    result = results["prefixItems"][0]
    assert result["properties"]["segmentId"] == {"const": "segment-1"}
    ranking = result["properties"]["speakerRanking"]
    assert ranking["minItems"] == ranking["maxItems"] == 2
    assert ranking["items"]["enum"] == ["speaker-1", "speaker-2"]
    assert result["properties"]["textEvidenceCandidateId"]["enum"] == [
        "",
        "nbest-2",
    ]
    assert result["properties"]["confidence"] == {"type": "number"}
    assert result["properties"]["evidenceRefs"]["items"]["enum"] == [
        "segment:segment-1",
        "speaker-score:segment-1:speaker-1",
        "speaker-score:segment-1:speaker-2",
        "asr-nbest:segment-1:nbest-2",
    ]


def test_prompt_token_timestamps_are_bounded_and_hash_bound() -> None:
    asr = {
        "provider": "fixture-asr",
        "timestamps": [
            {
                "text": f"token-{index}",
                "startMs": index * 10,
                "endMs": index * 10 + 5,
            }
            for index in range(80)
        ],
    }
    document = _document(first_asr=asr)

    request, refs = _segment_request(
        document,
        document["segments"],
        0,
        speaker_top_k=3,
    )

    assert len(request["tokenTimestamps"]) == 24
    assert request["tokenTimestamps"][0]["text"] == "token-0"
    assert request["tokenTimestamps"][-1]["text"] == "token-79"
    assert request["tokenTimestampEvidence"] == {
        "selectionMethod": "deterministic-even-sample-v1",
        "originalTokenCount": 80,
        "selectedTokenCount": 24,
        "sourceSha256": canonical_json_sha256(
            [
                {
                    "text": f"token-{index}",
                    "startMs": index * 10,
                    "endMs": index * 10 + 5,
                }
                for index in range(80)
            ]
        ),
    }
    assert "asr-tokens:segment-1" in refs


def test_semantic_runner_proposes_only_top_k_and_presentation_safe_text() -> None:
    document = _document()
    before = copy.deepcopy(document)
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    _result(
                        "segment-1",
                        ranking=["speaker-2", "speaker-1"],
                        text="I can go",
                        evidence_refs=[
                            "segment:segment-1",
                            "speaker-score:segment-1:speaker-2",
                        ],
                    ),
                    _result(
                        "segment-2",
                        ranking=["speaker-1", "speaker-2"],
                        text="Hello, world.",
                        evidence_refs=["segment:segment-2"],
                    ),
                    _keep("segment-3", "speaker-2", "Acknowledged"),
                ]
            }
        ]
    )

    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    assert document == before
    assert artifact["status"] == "completed"
    assert artifact["metrics"] == {
        "segmentsEvaluated": 3,
        "providerCalls": 1,
        "contextSplitCount": 0,
        "plannedBatchCount": 1,
        "plannedMaxBatchSize": 3,
        "contextTokenBudget": 3072,
        "maxEstimatedInputTokens": 2664,
        "suggestionCount": 2,
        "speakerSuggestionCount": 1,
        "textSuggestionCount": 1,
        "acceptedResultCount": 3,
        "abstentionCount": 1,
        "unresolvedSegmentCount": 0,
        "rejectionCount": 0,
        "failureCount": 0,
        "autoAppliedCount": 0,
    }
    assert artifact["suggestions"][0]["proposal"] == {
        "targetSpeakerId": "speaker-2"
    }
    assert artifact["suggestions"][1]["proposal"] == {
        "normalizedText": "Hello, world.",
        "displayText": "Hello, world.",
    }

    queue = attach_semantic_suggestions_to_review(
        document,
        _empty_queue(),
        artifact,
        artifact_path="/tmp/semantic-suggestions.v1.json",
    )
    assert [item["reasonCode"] for item in queue["items"]] == [
        "SEMANTIC_SPEAKER_SUGGESTION",
        "SEMANTIC_TEXT_SUGGESTION",
    ]
    assert queue["openCount"] == 2
    assert document == before


def test_token_budget_aware_packing_avoids_known_context_overflow() -> None:
    class BudgetRecordingProvider:
        provider_id = "budget-recording-fixture"
        provider_version = "1"
        network_policy = "loopback-only"
        config = LocalLLMConfig(context_tokens=4_096, output_tokens=1_024)

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def generate_json(self, **kwargs: object) -> dict[str, object]:
            payload = json.loads(str(kwargs["user_prompt"]).split("input=", 1)[1])
            batch = payload["segments"]
            self.batch_sizes.append(len(batch))
            return {
                "results": [
                    _keep(
                        item["segmentId"],
                        item["currentSpeakerId"],
                        item["currentNormalizedText"],
                    )
                    for item in batch
                ]
            }

    document = _document()
    for segment in document["segments"]:
        long_text = "word " * 160
        segment["rawText"] = long_text
        segment["normalizedText"] = long_text
        segment["displayText"] = long_text
    provider = BudgetRecordingProvider()

    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    assert artifact["status"] == "completed"
    assert provider.batch_sizes == [1, 1, 1]
    assert artifact["metrics"]["plannedBatchCount"] == 3
    assert artifact["metrics"]["plannedMaxBatchSize"] == 1
    assert artifact["metrics"]["contextTokenBudget"] == 3_072
    assert artifact["metrics"]["maxEstimatedInputTokens"] <= 3_072
    assert artifact["metrics"]["contextSplitCount"] == 0


def test_context_overflow_splits_batches_without_recording_false_failure() -> None:
    class ContextLimitedProvider:
        provider_id = "context-limited-fixture"
        provider_version = "1"
        network_policy = "loopback-only"

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def generate_json(self, **kwargs: object) -> dict:
            payload = json.loads(str(kwargs["user_prompt"]).split("input=", 1)[1])
            segments = payload["segments"]
            self.batch_sizes.append(len(segments))
            if len(segments) > 1:
                raise LocalLLMContextWindowError("fixture context overflow")
            item = segments[0]
            return {
                "results": [
                    _keep(
                        item["segmentId"],
                        item["currentSpeakerId"],
                        item["currentNormalizedText"],
                    )
                ]
            }

    provider = ContextLimitedProvider()
    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(_document())

    assert artifact["status"] == "completed"
    assert artifact["failures"] == []
    assert artifact["metrics"]["providerCalls"] == 3
    assert artifact["metrics"]["contextSplitCount"] == 2
    assert artifact["metrics"]["acceptedResultCount"] == 3
    assert provider.batch_sizes == [3, 2, 1, 1, 1]


def test_lexical_repair_requires_and_binds_exact_nbest_candidate() -> None:
    nbest = {
        "provider": "fixture-asr",
        "nBest": [
            {
                "candidateId": "nbest-2",
                "text": "I cannot go",
                "language": "en",
                "score": -0.2,
                "lexicalRepairEligible": True,
            }
        ],
    }
    document = _document(first_asr=nbest)
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    _result(
                        "segment-1",
                        ranking=["speaker-1", "speaker-2"],
                        text="I cannot go",
                        candidate_id="nbest-2",
                        evidence_refs=[
                            "segment:segment-1",
                            "asr-nbest:segment-1:nbest-2",
                        ],
                    ),
                    _keep("segment-2", "speaker-1", "Hello world"),
                    _keep("segment-3", "speaker-2", "Acknowledged"),
                ]
            }
        ]
    )

    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    assert artifact["metrics"]["textSuggestionCount"] == 1
    patch = artifact["suggestions"][0]["textPatch"]
    assert patch["lexicalChange"] is True
    assert patch["protectedTokenChange"] is True
    assert patch["evidenceCandidateId"] == "nbest-2"


def test_lexical_repair_rejects_ineligible_nbest_candidate() -> None:
    document = _document(
        first_asr={
            "provider": "fixture-asr",
            "nBest": [
                {
                    "candidateId": "nbest-2",
                    "text": "I cannot go",
                    "language": "en",
                    "score": -0.2,
                    "lexicalRepairEligible": False,
                }
            ],
        }
    )
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    _result(
                        "segment-1",
                        ranking=["speaker-1", "speaker-2"],
                        text="I cannot go",
                        candidate_id="nbest-2",
                        evidence_refs=[
                            "segment:segment-1",
                            "asr-nbest:segment-1:nbest-2",
                        ],
                    ),
                    _keep("segment-2", "speaker-1", "Hello world"),
                    _keep("segment-3", "speaker-2", "Acknowledged"),
                ]
            }
        ]
    )

    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    assert artifact["status"] == "partial"
    assert artifact["suggestions"] == []
    assert artifact["rejections"][0]["code"] == "SEMANTIC_TEXT_CHANGE_UNSUPPORTED"
    assert artifact["evidenceAvailability"] == {
        "segmentsWithAsrNBest": 1,
        "segmentsWithLexicalEligibleAlternatives": 0,
        "segmentsWithTokenTimestamps": 0,
        "segmentsWithLowSpeakerMargin": 3,
        "nativeSpeakerTimeline": False,
    }


def test_unsupported_lexical_change_and_human_lock_conflict_are_rejected() -> None:
    document = _document()
    document["segments"][0]["humanLocked"] = True
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    _result(
                        "segment-1",
                        ranking=["speaker-2", "speaker-1"],
                        text="I invented words",
                        evidence_refs=[
                            "segment:segment-1",
                            "speaker-score:segment-1:speaker-2",
                        ],
                    ),
                    _keep("segment-2", "speaker-1", "Hello world"),
                    _keep("segment-3", "speaker-2", "Acknowledged"),
                ]
            }
        ]
    )

    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    assert artifact["suggestions"] == []
    assert artifact["status"] == "partial"
    assert artifact["rejections"] == [
        {
            "segmentId": "segment-1",
            "code": "SEMANTIC_HUMAN_LOCK_CONFLICT",
            "message": "semantic proposal conflicts with a human speaker lock",
        }
    ]
    queue = attach_semantic_suggestions_to_review(
        document,
        _empty_queue(),
        artifact,
        artifact_path="/tmp/semantic-suggestions.v1.json",
    )
    assert queue["items"][0]["reasonCode"] == "SEMANTIC_PROCESSING_FAILED"
    assert queue["items"][0]["rejections"] == artifact["rejections"]


def test_invalid_confidence_rejects_only_the_affected_batch_result() -> None:
    invalid = _keep("segment-3", "speaker-2", "Acknowledged")
    invalid["confidence"] = 2.0
    artifact = SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(
            [
                {
                    "results": [
                        _keep("segment-1", "speaker-1", "I can go"),
                        _keep("segment-2", "speaker-1", "Hello world"),
                        invalid,
                    ]
                }
            ]
        ),
        model="fixture",
        batch_size=3,
    ).run(_document())

    assert artifact["status"] == "partial"
    assert artifact["failures"] == []
    assert artifact["metrics"]["acceptedResultCount"] == 2
    assert artifact["metrics"]["rejectionCount"] == 1
    assert artifact["rejections"] == [
        {
            "segmentId": "segment-3",
            "code": "SEMANTIC_RESPONSE_INVALID",
            "message": "semantic confidence must be finite and between 0 and 1",
        }
    ]


def test_provider_failure_is_durable_and_adds_a_review_blocker() -> None:
    document = _document()
    artifact = SemanticProcessingRunner(
        provider=MappingLocalLLMProvider([]),
        model="fixture",
        batch_size=3,
    ).run(document)

    assert artifact["status"] == "failed"
    assert artifact["metrics"]["failureCount"] == 1
    queue = attach_semantic_suggestions_to_review(
        document,
        _empty_queue(),
        artifact,
        artifact_path="/tmp/semantic-suggestions.v1.json",
    )
    assert queue["items"][0]["id"] == "semantic-processing-failure"
    assert queue["items"][0]["reasonCode"] == "SEMANTIC_PROCESSING_FAILED"


def test_semantic_artifact_rejects_transcript_rebinding_and_raw_text_patch() -> None:
    document = _document()
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    _result(
                        "segment-1",
                        ranking=["speaker-2", "speaker-1"],
                        text="I can go",
                        evidence_refs=[
                            "segment:segment-1",
                            "speaker-score:segment-1:speaker-2",
                        ],
                    ),
                    _keep("segment-2", "speaker-1", "Hello world"),
                    _keep("segment-3", "speaker-2", "Acknowledged"),
                ]
            }
        ]
    )
    artifact = SemanticProcessingRunner(
        provider=provider,
        model="fixture",
        batch_size=3,
    ).run(document)

    with pytest.raises(WorkerError, match="immutable transcript"):
        validate_semantic_suggestions_artifact(
            artifact,
            expected_job_id="semantic-fixture",
            expected_transcript_sha256="f" * 64,
        )

    tampered = copy.deepcopy(artifact)
    tampered["suggestions"][0]["proposal"]["rawText"] = "tampered"
    with pytest.raises(WorkerError, match="cannot contain rawText"):
        validate_semantic_suggestions_artifact(
            tampered,
            expected_job_id="semantic-fixture",
            expected_transcript_sha256=canonical_json_sha256(document),
        )
