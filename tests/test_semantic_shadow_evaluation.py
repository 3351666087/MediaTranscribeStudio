from __future__ import annotations

import copy

import pytest

from backend import MappingLocalLLMProvider, SemanticProcessingRunner
from tools.evaluate_semantic_shadow import build_shadow_report


def _segment(
    segment_id: str,
    *,
    speaker_id: str,
    text: str,
    start_ms: int,
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
        "humanLocked": False,
        "revisions": [],
        "language": "en",
        "evidence": {"asr": {"provider": "fixture-asr"}},
    }


def _document() -> dict:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-shadow-fixture",
        "jobId": "shadow-fixture",
        "generatedAt": "2026-07-28T00:00:00Z",
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
                text="alpha",
                start_ms=0,
            ),
            _segment(
                "segment-2",
                speaker_id="speaker-1",
                text="beta",
                start_ms=1_000,
            ),
            _segment(
                "segment-3",
                speaker_id="speaker-2",
                text="gamma",
                start_ms=2_000,
            ),
        ],
        "provenance": {"offline": True, "models": []},
    }


def _case() -> dict:
    turns = [
        {
            "startSeconds": 0.0,
            "endSeconds": 1.0,
            "speakerId": "truth-a",
        },
        {
            "startSeconds": 1.0,
            "endSeconds": 2.0,
            "speakerId": "truth-b",
        },
        {
            "startSeconds": 2.0,
            "endSeconds": 3.0,
            "speakerId": "truth-b",
        },
    ]
    transcript_turns = [
        {**turn, "transcript": text}
        for turn, text in zip(turns, ("alpha", "beta", "gamma"))
    ]
    return {
        "id": "shadow-case",
        "evaluationSplit": "held-out",
        "language": "en",
        "expectedSpeakerCount": 2,
        "scoringTranscript": "alpha beta gamma",
        "turns": turns,
        "referenceTranscriptTurns": transcript_turns,
        "languageTruth": {
            "qualification": "exact-fixture",
            "expectedLanguages": ["en"],
            "timeScoringEligible": True,
            "wordScoringEligible": True,
            "intervals": [
                {
                    "language": "en",
                    "startSeconds": turn["startSeconds"],
                    "endSeconds": turn["endSeconds"],
                    "speakerId": turn["speakerId"],
                    "transcript": text,
                }
                for turn, text in zip(turns, ("alpha", "beta", "gamma"))
            ],
        },
        "factualTruth": {
            "requiredLiterals": ["alpha", "beta"],
            "forbiddenLiterals": ["delta"],
        },
        "truthEligibility": {
            "speakerCount": True,
            "turnBoundaries": True,
            "derJer": True,
            "asr": True,
            "language": True,
        },
    }


def _artifact(*, propose: bool) -> dict:
    gate_results = [
        {
            "segmentId": "segment-1",
            "decision": "abstain",
            "confidence": 0.8,
        },
        {
            "segmentId": "segment-2",
            "decision": "propose" if propose else "abstain",
            "confidence": 0.9,
        },
        {
            "segmentId": "segment-3",
            "decision": "abstain",
            "confidence": 0.8,
        },
    ]
    responses = [{"results": gate_results}]
    if propose:
        responses.append(
            {
                "results": [
                    {
                        "segmentId": "segment-2",
                        "decision": "propose",
                        "speakerRanking": ["speaker-2", "speaker-1"],
                        "normalizedText": "beta",
                        "textEvidenceCandidateId": "",
                        "confidence": 0.9,
                        "reasonCodes": ["CONTEXTUAL_CANDIDATE_RERANK"],
                        "evidenceRefs": [
                            "segment:segment-2",
                            "speaker-score:segment-2:speaker-2",
                        ],
                    }
                ]
            }
        )
    return SemanticProcessingRunner(
        provider=MappingLocalLLMProvider(responses),
        model="fixture",
        batch_size=3,
    ).run(_document())


def test_shadow_apply_scores_validated_speaker_gain_without_mutation() -> None:
    document = _document()
    before = copy.deepcopy(document)

    report = build_shadow_report(
        case=_case(),
        document=document,
        semantic_artifact=_artifact(propose=True),
        review_open_count=2,
    )

    assert document == before
    assert report["overallOutcome"] == "improved"
    assert report["releaseApproved"] is False
    assert report["semantic"]["appliedSuggestionCount"] == 1
    assert (
        report["shadowMetrics"]["diarization"]["der"]
        < report["baselineMetrics"]["diarization"]["der"]
    )
    assert report["baselineMetrics"]["finalText"]["werOrCer"] == 0.0
    assert report["shadowMetrics"]["finalText"]["werOrCer"] == 0.0
    assert report["shadowMetrics"]["review"] == {
        "openCount": 2,
        "shadowResolutionsApplied": 0,
        "humanApprovalSimulated": False,
    }
    assert "requiredLiterals" not in report["shadowMetrics"]["factualIntegrity"]


def test_all_abstentions_report_no_measured_gain() -> None:
    report = build_shadow_report(
        case=_case(),
        document=_document(),
        semantic_artifact=_artifact(propose=False),
    )

    assert report["overallOutcome"] == "unchanged"
    assert report["semantic"]["appliedSuggestionCount"] == 0
    assert "no-measured-semantic-gain" in report["blockingReasons"]


def test_partial_semantic_artifact_is_not_shadow_applicable() -> None:
    artifact = _artifact(propose=False)
    artifact["status"] = "partial"

    with pytest.raises(ValueError, match="complete, unresolved-free"):
        build_shadow_report(
            case=_case(),
            document=_document(),
            semantic_artifact=artifact,
        )
