from __future__ import annotations

import copy
import hashlib

import pytest

from backend import (
    MappingLocalLLMProvider,
    SemanticProcessingRunner,
    WorkerError,
    build_final_adjudicated_transcript,
    build_final_no_speech_adjudication,
    validate_final_adjudicated_transcript,
    validate_final_no_speech_adjudication,
)
from backend.persistence import canonical_json_sha256
from backend.voice_activity import build_voice_activity


def _document() -> dict:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-final-fixture",
        "jobId": "final-fixture",
        "generatedAt": "2026-07-28T00:00:00Z",
        "language": "mul",
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
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "rawText": "Hello world",
                "normalizedText": "Hello, world.",
                "displayText": "Hello, world.",
                "confidence": 0.9,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.9},
                    {"speakerId": "speaker-2", "score": 0.1},
                ],
                "speakerMargin": 0.8,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [{"type": "text", "source": "manual"}],
                "language": "en",
                "evidence": {"asr": {"provider": "fixture-asr"}},
            },
            {
                "id": "segment-2",
                "startMs": 1_000,
                "endMs": 2_000,
                "speakerId": "speaker-2",
                "rawText": "你好",
                "normalizedText": "你好。",
                "displayText": "你好。",
                "confidence": 0.9,
                "speakerScores": [
                    {"speakerId": "speaker-1", "score": 0.1},
                    {"speakerId": "speaker-2", "score": 0.9},
                ],
                "speakerMargin": 0.8,
                "overlapping": True,
                "humanLocked": True,
                "revisions": [{"type": "text", "source": "manual"}],
                "language": "zh",
                "evidence": {"asr": {"provider": "fixture-asr"}},
            },
        ],
        "provenance": {"offline": True, "models": []},
    }


def _semantic(document: dict) -> dict:
    provider = MappingLocalLLMProvider(
        [
            {
                "results": [
                    {
                        "segmentId": "segment-1",
                        "decision": "abstain",
                        "confidence": 0.9,
                    },
                    {
                        "segmentId": "segment-2",
                        "decision": "abstain",
                        "confidence": 0.9,
                    },
                ]
            }
        ]
    )
    return SemanticProcessingRunner(
        provider=provider,
        model="qwen3.5:9b",
    ).run(document)


def _review_queue() -> dict:
    decisions = [
        {"decisionId": "decision-1"},
        {"decisionId": "decision-2"},
    ]
    return {
        "schemaVersion": "2.0.0",
        "jobId": "final-fixture",
        "openCount": 0,
        "items": [
            {
                "id": "review-1",
                "status": "accepted",
                "decision": decisions[0],
            },
            {
                "id": "review-2",
                "status": "rejected",
                "decision": decisions[1],
            },
        ],
        "decisions": decisions,
    }


def test_builds_hash_bound_final_scoring_subject() -> None:
    document = _document()
    semantic = _semantic(document)
    queue = _review_queue()

    final = build_final_adjudicated_transcript(document, queue, semantic)

    assert final["schemaVersion"] == "1.1.0"
    assert final["status"] == "adjudication-complete"
    assert final["disposition"] == "transcribable-speech"
    assert final["acceptanceSubject"] == "speaker-language-time-final-text"
    assert final["finalTextAuthority"] == "normalizedText"
    assert final["input"] == {
        "sourceMediaSha256": "a" * 64,
        "transcriptDocumentSha256": canonical_json_sha256(document),
        "semanticInputTranscriptSha256": canonical_json_sha256(document),
        "semanticArtifactSha256": canonical_json_sha256(semantic),
        "reviewQueueSha256": canonical_json_sha256(queue),
    }
    assert final["review"] == {
        "openCount": 0,
        "itemCount": 2,
        "acceptedCount": 1,
        "rejectedCount": 1,
        "decisionCount": 2,
    }
    assert [segment["finalText"] for segment in final["segments"]] == [
        "Hello, world.",
        "你好。",
    ]
    assert final["segments"][0]["rawTextSha256"] == hashlib.sha256(
        b"Hello world"
    ).hexdigest()
    assert final["segments"][1]["overlapping"] is True


def test_refuses_open_review_or_incomplete_semantic_state() -> None:
    document = _document()
    semantic = _semantic(document)
    queue = _review_queue()
    queue["items"][0]["status"] = "open"
    queue["openCount"] = 1
    with pytest.raises(
        WorkerError,
        match="final adjudication refuses open review items",
    ):
        build_final_adjudicated_transcript(document, queue, semantic)

    queue = _review_queue()
    semantic["metrics"]["autoAppliedCount"] = 1
    with pytest.raises(
        WorkerError,
        match="semantic processing is not complete and approval-safe",
    ):
        build_final_adjudicated_transcript(document, queue, semantic)


def test_validation_rejects_final_text_or_review_rebinding() -> None:
    document = _document()
    semantic = _semantic(document)
    queue = _review_queue()
    final = build_final_adjudicated_transcript(document, queue, semantic)

    tampered_text = copy.deepcopy(final)
    tampered_text["segments"][0]["finalText"] = "Invented"
    with pytest.raises(
        WorkerError,
        match="segments do not match the reviewed transcript",
    ):
        validate_final_adjudicated_transcript(
            tampered_text,
            expected_document=document,
            expected_review_queue=queue,
            expected_semantic_artifact=semantic,
        )

    changed_queue = copy.deepcopy(queue)
    changed_queue["decisions"][0]["decisionId"] = "rebound"
    with pytest.raises(
        WorkerError,
        match="source hashes do not match current evidence",
    ):
        validate_final_adjudicated_transcript(
            final,
            expected_document=document,
            expected_review_queue=changed_queue,
            expected_semantic_artifact=semantic,
        )


def test_builds_hash_bound_no_speech_final_subject_without_transcript() -> None:
    voice = build_voice_activity(
        job_id="no-speech-fixture",
        source_sha256="b" * 64,
        media_duration_ms=5_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "FunASR", "version": "1.2.0"},
        windows=(),
        minimum_window_ms=120,
        classification="no-speech-candidates-detected",
        has_transcribable_speech=False,
    )

    final = build_final_no_speech_adjudication(voice)

    assert final["schemaVersion"] == "1.1.0"
    assert final["disposition"] == "no-transcribable-speech"
    assert final["acceptanceSubject"] == "lexical-speech-presence"
    assert final["segments"] == []
    assert "documentId" not in final
    assert "semantic" not in final
    assert "review" not in final
    assert "speakerPolicy" not in final
    assert "finalTextAuthority" not in final
    assert final["input"] == {
        "sourceMediaSha256": "b" * 64,
        "voiceActivitySha256": canonical_json_sha256(voice),
    }

    tampered = copy.deepcopy(final)
    tampered["voiceActivity"]["speechDurationMs"] = 1
    with pytest.raises(
        WorkerError,
        match="summary does not match voice evidence",
    ):
        validate_final_no_speech_adjudication(
            tampered,
            expected_voice_activity=voice,
        )


def test_no_speech_final_refuses_transcribable_voice_activity() -> None:
    voice = build_voice_activity(
        job_id="speech-fixture",
        source_sha256="c" * 64,
        media_duration_ms=5_000,
        normalization_profile="mono-16khz-f32-v1",
        provider={"id": "FunASR", "version": "1.2.0"},
        windows=({"id": "vad-1", "startMs": 0, "endMs": 5_000},),
        minimum_window_ms=120,
        classification="transcribable-speech-detected",
        has_transcribable_speech=True,
    )

    with pytest.raises(
        WorkerError,
        match="requires a non-transcribable disposition",
    ):
        build_final_no_speech_adjudication(voice)
