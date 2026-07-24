from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    return manifest, results, outputs


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
    assert summary["gates"] == {
        "technicalExecutionPassed": True,
        "allCasesRequireReview": True,
        "languageWindowLimitPassed": True,
        "referenceQualityScored": False,
        "qualityConclusion": "not_scored_missing_reference_truth",
    }
    serialized = json.dumps(summary)
    assert "rawText" not in serialized
    assert "normalizedText" not in serialized
    assert "displayText" not in serialized


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
