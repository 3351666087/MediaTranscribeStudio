from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.compare_speaker_verification_reports import (
    SpeakerVerificationComparisonError,
    compare_reports,
)


def _report(
    path: Path,
    *,
    model: str,
    auc: float,
    eer: float,
    evaluation_split: str = "development",
) -> None:
    trials = [
        {
            "trialId": "same",
            "enrollmentClipId": "a-1",
            "testClipId": "a-2",
            "sameSpeaker": True,
            "cosineScore": 0.8,
        },
        {
            "trialId": "different",
            "enrollmentClipId": "a-1",
            "testClipId": "b-1",
            "sameSpeaker": False,
            "cosineScore": 0.2,
        },
    ]
    trial_manifest = path.parent / "frozen-trials.json"
    if not trial_manifest.exists():
        manifest = {
            "schemaVersion": "1.0.0",
            "source": {
                "recordingId": "recording-1",
                "audioSha256": "d" * 64,
                "annotationSha256": "e" * 64,
            },
            "clips": [
                {
                    "clipId": "a-1",
                    "speakerId": "a",
                    "evaluationSplit": evaluation_split,
                },
                {
                    "clipId": "a-2",
                    "speakerId": "a",
                    "evaluationSplit": evaluation_split,
                },
                {
                    "clipId": "b-1",
                    "speakerId": "b",
                    "evaluationSplit": evaluation_split,
                },
            ],
            "trials": [
                {key: value for key, value in trial.items() if key != "cosineScore"}
                for trial in trials
            ],
            "counts": {
                "speakers": 2,
                "clips": 3,
                "totalTrials": 2,
            },
        }
        manifest["canonicalSha256"] = canonical_json_sha256(manifest)
        trial_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    manifest = json.loads(trial_manifest.read_text(encoding="utf-8"))
    value = {
        "schemaVersion": "1.0.0",
        "benchmark": "frozen-speaker-verification-trials",
        "model": {
            "modelKey": model,
            "manifestFileSha256": ("a" if model == "baseline" else "b") * 64,
        },
        "trialManifest": {
            "path": str(trial_manifest),
            "fileSha256": sha256_file(trial_manifest),
            "canonicalSha256": manifest["canonicalSha256"],
        },
        "partition": {
            "evaluationSplit": evaluation_split,
            "sourceRecordingId": "recording-1",
            "recordingCount": 1,
            "speakerCount": 2,
            "clipCount": 3,
            "trialCount": 2,
        },
        "source": {
            "audioSha256": "d" * 64,
            "annotationSha256": "e" * 64,
        },
        "scores": {
            "rocAuc": auc,
            "equalErrorRate": eer,
            "minimumDetectionCostP01": eer / 10.0,
            "balancedAccuracyAtEerThreshold": 1.0 - eer,
        },
        "resources": {
            "peakCudaReservedMb": 1000.0,
            "retainedCudaAllocatedMb": 8.0,
        },
        "trials": trials,
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_quality_regression_is_rejected_before_held_out(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    challenger = tmp_path / "challenger.json"
    _report(baseline, model="baseline", auc=0.99, eer=0.05)
    _report(challenger, model="challenger", auc=0.98, eer=0.07)

    report = compare_reports(
        baseline_report_path=baseline,
        challenger_report_path=challenger,
        vram_limit_mb=12_288.0,
    )

    assert report["gates"]["qualityPassed"] is False
    assert report["gates"]["resourcesPassed"] is True
    assert report["decision"]["status"] == (
        "reject-development-or-regression-regression"
    )
    assert report["decision"]["productionPromotionAllowed"] is False
    canonical = dict(report)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)


def test_different_trial_identity_fails_closed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    challenger = tmp_path / "challenger.json"
    _report(baseline, model="baseline", auc=0.99, eer=0.05)
    _report(challenger, model="challenger", auc=0.99, eer=0.05)
    value = json.loads(challenger.read_text(encoding="utf-8"))
    value["trials"][0]["trialId"] = "other"
    value.pop("canonicalSha256")
    value["canonicalSha256"] = canonical_json_sha256(value)
    challenger.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        SpeakerVerificationComparisonError,
        match="scored trial identities differ",
    ):
        compare_reports(
            baseline_report_path=baseline,
            challenger_report_path=challenger,
            vram_limit_mb=12_288.0,
        )


def test_raw_held_out_eer_reports_cannot_claim_promotion(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    challenger = tmp_path / "challenger.json"
    _report(
        baseline,
        model="baseline",
        auc=0.98,
        eer=0.08,
        evaluation_split="held-out",
    )
    _report(
        challenger,
        model="challenger",
        auc=0.99,
        eer=0.04,
        evaluation_split="held-out",
    )

    with pytest.raises(
        SpeakerVerificationComparisonError,
        match="development-frozen threshold evidence",
    ):
        compare_reports(
            baseline_report_path=baseline,
            challenger_report_path=challenger,
            vram_limit_mb=12_288.0,
        )


def test_redimnet_quality_win_requires_runtime_release_evidence(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    challenger = tmp_path / "redimnet.json"
    _report(baseline, model="baseline", auc=0.98, eer=0.08)
    manifest_path = tmp_path / "frozen-trials.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_trials = manifest["trials"]
    value = {
        "schemaVersion": "1.0.0",
        "benchmark": "redimnet2-frozen-speaker-verification",
        "model": {
            "repoId": "Wespeaker/redimnet2",
            "manifestFileSha256": "b" * 64,
        },
        "trialManifest": {
            "path": str(manifest_path),
            "fileSha256": sha256_file(manifest_path),
            "canonicalSha256": manifest["canonicalSha256"],
            "evaluationSplit": "development",
            "sourceAudioSha256": "d" * 64,
            "sourceAnnotationSha256": "e" * 64,
        },
        "execution": {
            "speakerCount": 2,
            "clipCount": 3,
            "trialCount": 2,
            "peakReservedVramBytes": 4_000 * 1024 * 1024,
        },
        "metrics": {
            "auc": 0.99,
            "eer": 0.04,
            "minDcfPTarget0.01": 0.001,
            "balancedAccuracyAtEer": 0.96,
        },
        "trialScores": [
            {**trial, "cosine": 0.8 if trial["sameSpeaker"] else 0.2}
            for trial in base_trials
        ],
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    challenger.write_text(json.dumps(value), encoding="utf-8")

    report = compare_reports(
        baseline_report_path=baseline,
        challenger_report_path=challenger,
        vram_limit_mb=12_288.0,
    )

    assert report["gates"]["qualityPassed"] is True
    assert report["gates"]["resourcesPassed"] is False
    assert report["gates"]["candidateAdvancementAllowed"] is True
    assert report["decision"]["status"] == (
        "advance-to-runtime-release-validation"
    )
    assert report["decision"]["productionPromotionAllowed"] is False
