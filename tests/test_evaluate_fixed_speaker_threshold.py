from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.evaluate_fixed_speaker_threshold import (
    FixedSpeakerThresholdError,
    evaluate_fixed_threshold,
)


def _manifest(path: Path, *, split: str, speakers: tuple[str, str]) -> Path:
    clips = []
    for speaker in speakers:
        for index in (1, 2):
            clips.append(
                {
                    "clipId": f"{speaker}-{index}",
                    "speakerId": speaker,
                    "originalRecordingId": f"{speaker}-recording-{index}",
                    "evaluationSplit": split,
                }
            )
    trials = [
        {
            "trialId": "target-a",
            "enrollmentClipId": f"{speakers[0]}-1",
            "testClipId": f"{speakers[0]}-2",
            "sameSpeaker": True,
            "evaluationSplit": split,
        },
        {
            "trialId": "target-b",
            "enrollmentClipId": f"{speakers[1]}-1",
            "testClipId": f"{speakers[1]}-2",
            "sameSpeaker": True,
            "evaluationSplit": split,
        },
        {
            "trialId": "impostor-a",
            "enrollmentClipId": f"{speakers[0]}-1",
            "testClipId": f"{speakers[1]}-1",
            "sameSpeaker": False,
            "evaluationSplit": split,
        },
        {
            "trialId": "impostor-b",
            "enrollmentClipId": f"{speakers[0]}-2",
            "testClipId": f"{speakers[1]}-2",
            "sameSpeaker": False,
            "evaluationSplit": split,
        },
    ]
    value = {
        "schemaVersion": "1.0.0",
        "source": {"dataset": "openslr/82/cn-celeb-v2"},
        "clips": clips,
        "trials": trials,
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _report(
    path: Path,
    *,
    manifest: Path,
    scores: tuple[float, float, float, float],
    threshold: float = 0.5,
) -> Path:
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    trials = [
        {**trial, "cosineScore": score}
        for trial, score in zip(frozen["trials"], scores)
    ]
    value = {
        "schemaVersion": "1.0.0",
        "benchmark": "frozen-speaker-verification-trials",
        "model": {
            "modelKey": "fixture-model",
            "manifestFileSha256": "a" * 64,
        },
        "trialManifest": {
            "path": str(manifest),
            "fileSha256": sha256_file(manifest),
            "canonicalSha256": frozen["canonicalSha256"],
        },
        "scores": {
            "equalErrorThreshold": threshold,
            "equalErrorRate": 0.25,
            "rocAuc": 0.75,
        },
        "trials": trials,
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_development_threshold_is_applied_without_held_out_refit(
    tmp_path: Path,
) -> None:
    development_manifest = _manifest(
        tmp_path / "development.json",
        split="development",
        speakers=("dev-a", "dev-b"),
    )
    held_out_manifest = _manifest(
        tmp_path / "held-out.json",
        split="held-out",
        speakers=("held-a", "held-b"),
    )
    development = _report(
        tmp_path / "development-report.json",
        manifest=development_manifest,
        scores=(0.9, 0.8, 0.1, 0.2),
    )
    held_out = _report(
        tmp_path / "held-out-report.json",
        manifest=held_out_manifest,
        scores=(0.7, 0.4, 0.6, 0.2),
        threshold=0.9,
    )

    result = evaluate_fixed_threshold(
        development_report_path=development,
        held_out_report_path=held_out,
    )

    metrics = result["heldOut"]["metricsAtDevelopmentFrozenThreshold"]
    assert result["thresholdPolicy"]["threshold"] == 0.5
    assert result["thresholdPolicy"]["heldOutThresholdFittingPerformed"] is False
    assert metrics["falseRejectRate"] == 0.5
    assert metrics["falseAcceptRate"] == 0.5
    assert metrics["balancedAccuracy"] == 0.5
    assert result["heldOut"]["descriptiveOnly"][
        "promotionDecisionUsesHeldOutEerThreshold"
    ] is False
    canonical = dict(result)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)


def test_speaker_overlap_between_splits_fails_closed(tmp_path: Path) -> None:
    development_manifest = _manifest(
        tmp_path / "development.json",
        split="development",
        speakers=("same", "dev-b"),
    )
    held_out_manifest = _manifest(
        tmp_path / "held-out.json",
        split="held-out",
        speakers=("same", "held-b"),
    )
    development = _report(
        tmp_path / "development-report.json",
        manifest=development_manifest,
        scores=(0.9, 0.8, 0.1, 0.2),
    )
    held_out = _report(
        tmp_path / "held-out-report.json",
        manifest=held_out_manifest,
        scores=(0.9, 0.8, 0.1, 0.2),
    )

    with pytest.raises(FixedSpeakerThresholdError, match="not disjoint"):
        evaluate_fixed_threshold(
            development_report_path=development,
            held_out_report_path=held_out,
        )


def test_same_recording_trial_fails_closed(tmp_path: Path) -> None:
    development_manifest = _manifest(
        tmp_path / "development.json",
        split="development",
        speakers=("dev-a", "dev-b"),
    )
    value = json.loads(development_manifest.read_text(encoding="utf-8"))
    value["clips"][1]["originalRecordingId"] = value["clips"][0][
        "originalRecordingId"
    ]
    value.pop("canonicalSha256")
    value["canonicalSha256"] = canonical_json_sha256(value)
    development_manifest.write_text(json.dumps(value), encoding="utf-8")
    held_out_manifest = _manifest(
        tmp_path / "held-out.json",
        split="held-out",
        speakers=("held-a", "held-b"),
    )
    development = _report(
        tmp_path / "development-report.json",
        manifest=development_manifest,
        scores=(0.9, 0.8, 0.1, 0.2),
    )
    held_out = _report(
        tmp_path / "held-out-report.json",
        manifest=held_out_manifest,
        scores=(0.9, 0.8, 0.1, 0.2),
    )

    with pytest.raises(FixedSpeakerThresholdError, match="cross-recording"):
        evaluate_fixed_threshold(
            development_report_path=development,
            held_out_report_path=held_out,
        )
