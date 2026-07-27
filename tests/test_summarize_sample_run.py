from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import summarize_sample_run
from tools.summarize_sample_run import summarize_run


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    manifest = root / "manifest.json"
    results = root / "results"
    outputs = root / "outputs"
    case_id = "sample-a"
    job_id = f"sample-{case_id}"
    _write(
        manifest,
        {
            "libraryId": "fixture-library",
            "cases": [
                {
                    "id": case_id,
                    "sourceId": "source-a",
                    "sha256": "a" * 64,
                    "windowSelection": {
                        "reason": "stratum-middle",
                        "startMs": 1000,
                        "endMs": 14000,
                        "durationMs": 13000,
                        "audioActivityRatio": 0.8,
                    },
                    "truthEligibility": {
                        "speakerCount": False,
                        "asr": False,
                    },
                }
            ],
        },
    )
    _write(
        results / f"{case_id}-result.json",
        {
            "status": "observed",
            "terminal_type": "review.required",
            "job_id": job_id,
            "exit_code": 0,
            "elapsed_seconds": 4.5,
            "shutdown_acknowledged": True,
            "forced_cleanup_pids": [],
        },
    )
    output = outputs / case_id
    _write(
        output / "voice-activity.v1.json",
        {
            "jobId": job_id,
            "sourceSha256": "a" * 64,
            "classification": "transcribable-speech-detected",
            "mediaDurationMs": 13000,
            "speechWindowCount": 1,
            "speechDurationMs": 12500,
            "speechRatio": 0.961538462,
        },
    )
    refinement = {
        "maxLanguageWindowMs": 12000,
        "speakerChangeSplitsMs": [6000],
        "languageDurationSplitsMs": [12000],
        "appliedSplitsMs": [6000, 12000],
    }
    _write(
        output / "transcript-document.v2.json",
        {
            "jobId": job_id,
            "language": "mul",
            "segments": [
                {
                    "startMs": 0,
                    "endMs": 6000,
                    "speakerId": "speaker-1",
                    "evidence": {
                        "asr": {"language": "en"},
                        "speakerChangeRefinement": refinement,
                    },
                },
                {
                    "startMs": 6000,
                    "endMs": 12000,
                    "speakerId": "speaker-2",
                    "evidence": {
                        "asr": {"language": "zh"},
                        "speakerChangeRefinement": refinement,
                    },
                },
            ],
        },
    )
    _write(
        output / "review" / "review-queue.json",
        {
            "jobId": job_id,
            "openCount": 2,
            "items": [
                {
                    "status": "open",
                    "reasonCode": "SPEAKER_COUNT_LOW_CONFIDENCE",
                },
                {"status": "open", "reasonCode": "SEGMENT_LOW_CONFIDENCE"},
            ],
            "speakerCountEstimate": {
                "estimatedCount": 2,
                "confidence": 0.6,
                "candidateRange": {"min": 1, "max": 3},
                "method": "fixture",
            },
        },
    )
    _write(
        output / "pipeline-metrics.v1.json",
        {
            "jobId": job_id,
            "runtime": {"elapsedMs": 2000, "rtf": 0.153846154},
            "resources": {"peakRamMb": 256, "peakVramMb": None},
            "cache": {
                "requests": 4,
                "hitRate": 0.5,
                "recomputationRate": 0.0,
            },
            "routing": {"escalationRate": 0.25},
            "policy": {
                "resolvedSpeakerCount": 2,
                "speakerCountMode": "auto",
            },
            "offline": True,
            "referenceEvaluation": {"available": False},
        },
    )
    _write(
        output / "semantic" / "semantic-suggestions.v1.json",
        {
            "jobId": job_id,
            "status": "completed",
            "model": "qwen3.5:9b",
            "promptVersion": "semantic-fixture-v1",
            "applicationPolicy": "suggestion-only",
            "requiresHumanApproval": True,
            "provider": {
                "id": "ollama-loopback",
                "version": "fixture",
                "networkPolicy": "loopback-only",
            },
            "metrics": {
                "segmentsEvaluated": 2,
                "providerCalls": 2,
                "gateProviderCalls": 2,
                "proposalProviderCalls": 0,
                "acceptedResultCount": 2,
                "abstentionCount": 2,
                "suggestionCount": 0,
                "speakerSuggestionCount": 0,
                "textSuggestionCount": 0,
                "rejectionCount": 0,
                "failureCount": 0,
                "unresolvedSegmentCount": 0,
                "autoAppliedCount": 0,
                "providerCompletedCalls": 2,
                "providerTotalDurationNanoseconds": 2_000_000_000,
                "providerLoadDurationNanoseconds": 200_000_000,
                "providerPromptEvalTokens": 100,
                "providerPromptEvalDurationNanoseconds": 1_000_000_000,
                "providerOutputTokens": 20,
                "providerOutputEvalDurationNanoseconds": 800_000_000,
            },
        },
    )
    return manifest, results, outputs


def test_rejects_dataless_manifest_before_json_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    monkeypatch.setattr(
        summarize_sample_run,
        "_path_is_dataless",
        lambda path: path == manifest,
    )

    with pytest.raises(ValueError, match="dataless cloud placeholder"):
        summarize_run(
            manifest_path=manifest,
            results_root=results,
            outputs_root=outputs,
        )


def test_summarizes_content_free_audit_and_separate_gates(
    tmp_path: Path,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)

    summary = summarize_run(
        manifest_path=manifest,
        results_root=results,
        outputs_root=outputs,
    )

    assert summary["privacy"] == {
        "containsTranscriptText": False,
        "containsSourceMediaPaths": False,
    }
    assert summary["aggregate"]["failedCases"] == 0
    assert summary["aggregate"]["reviewOpenCount"] == 2
    assert summary["aggregate"]["documentLanguageCounts"] == {"mul": 1}
    assert summary["aggregate"]["segmentLanguageCounts"] == {"en": 1, "zh": 1}
    assert summary["aggregate"]["maxObservedLanguageWindowMs"] == 6000
    assert summary["aggregate"]["semantic"] == {
        "evidenceCaseCount": 1,
        "statusCounts": {"completed": 1},
        "modelCounts": {"qwen3.5:9b": 1},
        "segmentsEvaluatedTotal": 2,
        "providerCallsTotal": 2,
        "acceptedResultCount": 2,
        "abstentionCount": 2,
        "suggestionCount": 0,
        "speakerSuggestionCount": 0,
        "textSuggestionCount": 0,
        "rejectionCount": 0,
        "failureCount": 0,
        "unresolvedSegmentCount": 0,
        "autoAppliedCount": 0,
        "providerTotalDurationSeconds": 2.0,
        "providerLoadDurationSeconds": 0.2,
        "providerPromptEvalDurationSeconds": 1.0,
        "providerOutputEvalDurationSeconds": 0.8,
        "providerPromptEvalTokens": 100,
        "providerOutputTokens": 20,
    }
    assert summary["gates"] == {
        "technicalExecutionPassed": True,
        "allCasesRequireReview": True,
        "languageWindowLimitPassed": True,
        "referenceQualityScored": False,
        "mandatorySemanticCompleted": True,
        "semanticSuggestionOnlyPolicyPassed": True,
        "lexicalSpeechNegativeGatePassed": None,
        "qualityConclusion": "not_scored_missing_reference_truth",
    }
    serialized = json.dumps(summary)
    assert "rawText" not in serialized
    assert "normalizedText" not in serialized
    assert "displayText" not in serialized


def test_summarizes_long_media_source_terminal_without_overclaiming(
    tmp_path: Path,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["selectionPolicy"] = {
        "requiredStrata": [
            "stratum-start",
            "stratum-middle",
            "stratum-end",
            "seeded-active-random",
            "acoustic-change",
        ],
        "modelScoresUsed": False,
    }
    manifest_value["sources"] = [
        {
            "id": "source-a",
            "sha256": "b" * 64,
            "durationMs": 14_000,
            "windowCount": 1,
            "windowCoverageRatio": 13 / 14,
            "analysis": {
                "algorithm": "fixture-full-timeline",
                "frameCount": 14,
                "frameDurationMs": 1000,
                "activeFrameRatio": 0.75,
                "activityThresholdDb": -30.0,
                "rmsDbPercentiles": {
                    "p05": -42.0,
                    "p50": -20.0,
                    "p95": -12.0,
                },
            },
        }
    ]
    _write(manifest, manifest_value)

    summary = summarize_run(
        manifest_path=manifest,
        results_root=results,
        outputs_root=outputs,
    )

    assert summary["schemaVersion"] == "1.2.0"
    source = summary["sourceSummaries"][0]
    assert source["sourceSha256"] == "b" * 64
    assert source["fullTimelineAcousticScan"] == {
        "available": True,
        "algorithm": "fixture-full-timeline",
        "frameCount": 14,
        "frameDurationMs": 1000,
        "analyzedDurationMs": 14_000,
        "coverageRatio": 1.0,
        "activeFrameRatio": 0.75,
        "activityThresholdDb": -30.0,
        "rmsDbPercentiles": {
            "p05": -42.0,
            "p50": -20.0,
            "p95": -12.0,
        },
        "isSpeechClassification": False,
    }
    assert source["stratifiedWindows"]["requiredStrataCovered"] == {
        "stratum-start": False,
        "stratum-middle": True,
        "stratum-end": False,
        "seeded-active-random": False,
        "acoustic-change": False,
    }
    assert source["speakerCountStability"]["distribution"] == {"2": 1}
    assert source["semantic"]["providerLoadDurationSeconds"] == 0.2
    assert source["semantic"]["providerPromptEvalTokens"] == 100
    assert source["terminal"] == {
        "windowTechnicalExecutionPassed": True,
        "mandatorySemanticEvidenceComplete": True,
        "fullTimelineAcousticScanPassed": True,
        "requiredStrataCovered": False,
        "windowSetComplete": True,
        "completeSourceProductionRunObserved": False,
        "speakerCountStableAcrossWindows": True,
        "referenceQualityScored": False,
        "manualFiveQualityEligible": False,
        "releaseApproved": False,
        "qualityConclusion": "not_scored_missing_reference_truth",
        "disposition": "review.required",
    }


def test_rejects_language_window_over_configured_limit(
    tmp_path: Path,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    transcript = outputs / "sample-a" / "transcript-document.v2.json"
    value = json.loads(transcript.read_text(encoding="utf-8"))
    value["segments"][1]["endMs"] = 19000
    _write(transcript, value)

    with pytest.raises(ValueError, match="language window exceeds"):
        summarize_run(
            manifest_path=manifest,
            results_root=results,
            outputs_root=outputs,
        )


def test_rejects_cross_job_artifact_link(
    tmp_path: Path,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    voice = outputs / "sample-a" / "voice-activity.v1.json"
    value = json.loads(voice.read_text(encoding="utf-8"))
    value["jobId"] = "sample-other"
    _write(voice, value)

    with pytest.raises(ValueError, match="jobId mismatch"):
        summarize_run(
            manifest_path=manifest,
            results_root=results,
            outputs_root=outputs,
        )


def test_rejects_cross_job_semantic_artifact_link(tmp_path: Path) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    semantic = (
        outputs
        / "sample-a"
        / "semantic"
        / "semantic-suggestions.v1.json"
    )
    value = json.loads(semantic.read_text(encoding="utf-8"))
    value["jobId"] = "sample-other"
    _write(semantic, value)

    with pytest.raises(ValueError, match="semantic jobId mismatch"):
        summarize_run(
            manifest_path=manifest,
            results_root=results,
            outputs_root=outputs,
        )


def test_summarizes_no_speech_negative_without_transcript(
    tmp_path: Path,
) -> None:
    manifest, results, outputs = _fixture(tmp_path)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    case = manifest_value["cases"][0]
    case["expectedLexicalSpeech"] = False
    case["selection"] = {
        "startSeconds": 1,
        "durationSeconds": 5,
    }
    del case["windowSelection"]
    _write(manifest, manifest_value)
    result_path = results / "sample-a-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["terminal_type"] = "job.completed"
    _write(result_path, result)
    output = outputs / "sample-a"
    voice_path = output / "voice-activity.v1.json"
    voice = json.loads(voice_path.read_text(encoding="utf-8"))
    voice["classification"] = "no-speech-candidates-detected"
    voice["speechWindowCount"] = 0
    voice["speechDurationMs"] = 0
    voice["speechRatio"] = 0
    _write(voice_path, voice)
    (output / "transcript-document.v2.json").unlink()
    (output / "pipeline-metrics.v1.json").unlink()
    (output / "review" / "review-queue.json").unlink()
    (output / "semantic" / "semantic-suggestions.v1.json").unlink()

    summary = summarize_run(
        manifest_path=manifest,
        results_root=results,
        outputs_root=outputs,
    )

    assert summary["cases"][0]["windowSelection"] == {
        "reason": None,
        "startMs": 1000,
        "endMs": 6000,
        "durationMs": 5000,
        "audioActivityRatio": None,
    }
    assert summary["aggregate"]["lexicalSpeechNegativeCases"] == 1
    assert summary["aggregate"]["lexicalSpeechFalsePositiveCount"] == 0
    assert summary["aggregate"]["lexicalSpeechFalsePositiveRate"] == 0
    assert summary["gates"]["lexicalSpeechNegativeGatePassed"] is True
    assert summary["gates"]["mandatorySemanticCompleted"] is None
    assert summary["gates"]["semanticSuggestionOnlyPolicyPassed"] is None
    assert (
        summary["gates"]["qualityConclusion"]
        == "lexical_speech_negative_gate_passed"
    )
