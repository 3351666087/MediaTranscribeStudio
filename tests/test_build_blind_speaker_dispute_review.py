from __future__ import annotations

import json
import wave
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.build_blind_speaker_dispute_review import (
    BlindSpeakerReviewError,
    build_review_package,
)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _canonical(value: dict) -> dict:
    return {**value, "canonicalSha256": canonical_json_sha256(value)}


def _fixture(root: Path) -> dict[str, object]:
    gate = root / "gate"
    report_root = gate / "reports"
    source_audio = gate / "development.composite.wav"
    annotation = _write_json(gate / "development.annotations.json", {"turns": []})
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    all_frames = bytearray()
    clip_frames = 1_600
    genres = ("interview", "singing", "speech", "drama", "movie", "vlog")
    originals = root / "originals"
    originals.mkdir()
    for index in range(36):
        pcm = int(index * 700 - 12_000).to_bytes(2, "little", signed=True)
        frames = pcm * clip_frames
        start_ms = index * 100
        all_frames.extend(frames)
        original = originals / f"source-{index:02d}.flac"
        original.write_bytes(frames)
        clips.append(
            {
                "audio": str(source_audio),
                "clipId": f"development-clip-{index:02d}",
                "startMs": start_ms,
                "endMs": start_ms + 100,
                "evaluationSplit": "development",
                "genre": genres[index % len(genres)],
                "originalRecordingId": f"recording-{index:02d}",
                "sourceFileBytes": original.stat().st_size,
                "sourceFilePath": str(original),
                "sourceFileSha256": sha256_file(original),
                "sourceRecordingId": "development-composite",
                "speakerId": f"speaker-{index // 2:02d}",
            }
        )
    with wave.open(str(source_audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(bytes(all_frames))

    trials: list[dict] = []
    for index in range(18):
        same_left = index * 2
        same_right = same_left + 1
        trials.append(
            {
                "trialId": f"development-trial-{len(trials):02d}",
                "enrollmentClipId": clips[same_left]["clipId"],
                "testClipId": clips[same_right]["clipId"],
                "sameSpeaker": True,
                "evaluationSplit": "development",
            }
        )
        different_right = ((index + 1) % 18) * 2
        trials.append(
            {
                "trialId": f"development-trial-{len(trials):02d}",
                "enrollmentClipId": clips[same_left]["clipId"],
                "testClipId": clips[different_right]["clipId"],
                "sameSpeaker": False,
                "evaluationSplit": "development",
            }
        )
    manifest_body = {
        "schemaVersion": "1.0.0",
        "libraryId": "fixture-development",
        "source": {
            "split": "development",
            "recordingId": "development-composite",
            "audioPath": str(source_audio),
            "audioBytes": source_audio.stat().st_size,
            "audioSha256": sha256_file(source_audio),
            "annotationPath": str(annotation),
            "annotationBytes": annotation.stat().st_size,
            "annotationSha256": sha256_file(annotation),
            "dataset": "fixture/dataset",
            "revision": "v1",
        },
        "counts": {
            "speakers": 18,
            "clips": len(clips),
            "totalTrials": len(trials),
        },
        "clips": clips,
        "trials": trials,
    }
    manifest = _canonical(manifest_body)
    manifest_path = _write_json(gate / "development.trials.json", manifest)

    report_paths: list[Path] = []
    fixed_paths: list[Path] = []
    model_names = ("secret-camp-model", "secret-wide-model")
    repo_ids = ("secret/camp-repository", "secret/wide-repository")
    for model_index, (model_name, repo_id) in enumerate(zip(model_names, repo_ids)):
        scored = []
        for trial_index, trial in enumerate(trials):
            if trial_index < 12:
                first = 0.90 if trial_index % 4 < 2 else 0.10
                second = 0.10 if trial_index % 4 < 2 else 0.90
            elif trial_index < 24:
                first = 0.62 if trial_index % 4 < 2 else 0.38
                second = 0.38 if trial_index % 4 < 2 else 0.62
            else:
                first = 0.501 if trial_index % 4 < 2 else 0.499
                second = 0.80 if trial_index % 4 < 2 else 0.20
            score = first if model_index == 0 else second
            scored.append({**trial, "cosineScore": score})
        model = {
            "path": str(root / model_name),
            "manifestPath": str(root / model_name / ".manifest.json"),
            "manifestFileSha256": ("a" if model_index == 0 else "b") * 64,
            "manifestCanonicalSha256": ("a" if model_index == 0 else "b") * 64,
            "modelKey": model_name,
            "provider": "secret-provider",
            "repoId": repo_id,
            "revision": "secret-revision",
        }
        report_body = {
            "schemaVersion": "1.0.0",
            "benchmark": "frozen-speaker-verification-trials",
            "model": model,
            "trialManifest": {
                "path": str(manifest_path),
                "fileSha256": sha256_file(manifest_path),
                "canonicalSha256": manifest["canonicalSha256"],
                "libraryId": manifest["libraryId"],
            },
            "partition": {
                "evaluationSplit": "development",
                "sourceRecordingId": "development-composite",
                "speakerCount": 18,
                "clipCount": len(clips),
                "trialCount": len(trials),
            },
            "source": {
                "audioSha256": sha256_file(source_audio),
                "annotationSha256": sha256_file(annotation),
            },
            "scores": {"equalErrorThreshold": 0.5},
            "trials": scored,
        }
        report = _canonical(report_body)
        report_path = _write_json(
            report_root / f"{model_name}.development.v1.json", report
        )
        fixed_body = {
            "schemaVersion": "1.0.0",
            "evaluation": "development-frozen-speaker-threshold",
            "model": model,
            "thresholdPolicy": {
                "sourceSplit": "development",
                "threshold": 0.5,
                "heldOutThresholdFittingPerformed": False,
            },
            "development": {
                "reportPath": str(report_path),
                "reportFileSha256": sha256_file(report_path),
                "reportCanonicalSha256": report["canonicalSha256"],
                "trialManifestCanonicalSha256": manifest["canonicalSha256"],
            },
        }
        fixed = _canonical(fixed_body)
        fixed_path = _write_json(
            report_root / f"{model_name}.fixed-threshold.v1.json", fixed
        )
        report_paths.append(report_path)
        fixed_paths.append(fixed_path)

    license_lock = _write_json(
        root / "dataset.lock.json",
        {
            "datasetKey": "fixture-dataset",
            "source": {"catalogUrl": "https://example.invalid/dataset"},
            "license": {"spdx": "CC-BY-SA-4.0"},
        },
    )
    readme = root / "README.TXT"
    readme.write_text(
        "License\nCN-Celeb is available for research. "
        "No commercial usage is permitted.\n",
        encoding="utf-8",
    )
    return {
        "reports": report_paths,
        "fixed": fixed_paths,
        "license": license_lock,
        "readme": readme,
        "manifest": manifest_path,
        "modelNames": model_names,
        "repoIds": repo_ids,
    }


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_package_is_blind_balanced_hash_bound_and_unblindable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    result = build_review_package(
        development_report_paths=fixture["reports"],
        fixed_threshold_report_paths=fixture["fixed"],
        output_root=tmp_path / "packages",
        license_lock_path=fixture["license"],
        dataset_readme_path=fixture["readme"],
        case_count=18,
        random_seed=b"A" * 32,
    )

    reviewer = Path(result["reviewerPacket"])
    review_manifest = _load(reviewer / "phase-1.review-manifest.v1.json")
    opinions = _load(reviewer / "phase-2.candidate-opinions.v1.json")
    vault = _load(result["identityMapping"])
    assert review_manifest["counts"] == {"cases": 18, "audioFilesPerCase": 3}
    assert review_manifest["blindnessPolicy"] == {
        "modelIdentityIncluded": False,
        "speakerIdentityIncluded": False,
        "referenceTruthIncluded": False,
        "automaticScoresIncluded": False,
        "confidenceIncluded": False,
        "sourcePathsIncluded": False,
    }
    body = dict(review_manifest)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
    assert review_manifest["evidenceCommitments"]["identityVaultFileSha256"] == (
        sha256_file(Path(result["identityMapping"]))
    )
    assert review_manifest["licenseEvidence"]["termsConflictRequiresLegalReview"]
    assert review_manifest["licenseEvidence"]["packageUse"] == (
        "local-research-review-only-do-not-redistribute"
    )

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in reviewer.rglob("*")
        if path.is_file() and path.suffix != ".wav"
    )
    for secret in (*fixture["modelNames"], *fixture["repoIds"]):
        assert secret not in public_text
    assert "development-trial-" not in public_text
    assert '"sameSpeaker"' not in public_text
    assert '"cosineScore"' not in public_text
    assert '"threshold"' not in public_text
    assert "secret-provider" not in public_text
    assert "secret-revision" not in public_text

    mapping_by_case = {item["caseId"]: item for item in vault["cases"]}
    assert set(mapping_by_case) == {item["caseId"] for item in review_manifest["cases"]}
    assert Counter(item["selectionBucket"] for item in vault["cases"]) == {
        "high-confidence-conflict": 6,
        "prediction-disagreement": 6,
        "threshold-near": 6,
    }
    for bucket in (
        "high-confidence-conflict",
        "prediction-disagreement",
        "threshold-near",
    ):
        rows = [item for item in vault["cases"] if item["selectionBucket"] == bucket]
        assert sum(item["sameSpeaker"] for item in rows) == 3
    assert len({item["genrePair"] for item in vault["cases"]}) >= 5
    assert len({speaker for item in vault["cases"] for speaker in item["speakerTuple"]}) >= 10
    assert all("model" in option for case in vault["cases"] for option in case["candidateOptionMapping"])
    assert all(
        len(case["candidateOptionMapping"]) == 2 and len(case["audioSideMapping"]) == 2
        for case in vault["cases"]
    )
    assert len(opinions["cases"]) == 18

    for case in review_manifest["cases"]:
        for audio in case["audio"]:
            path = reviewer / audio["path"]
            assert sha256_file(path) == audio["sha256"]
            with wave.open(str(path), "rb") as handle:
                assert handle.getparams()[:3] == (1, 2, 16_000)
                assert handle.getnframes() == 1_600
        pair = reviewer / case["combinedPlayback"]["path"]
        assert sha256_file(pair) == case["combinedPlayback"]["sha256"]
        with wave.open(str(pair), "rb") as handle:
            assert handle.getnframes() == 1_600 + 12_000 + 1_600

    with zipfile.ZipFile(result["reviewerZip"]) as archive:
        names = archive.namelist()
        assert names
        assert all(name.startswith("reviewer-packet/") for name in names)
        assert not any("identity-vault" in name for name in names)
        assert not any("DO-NOT-SHARE" in name for name in names)


def test_secure_seed_randomizes_case_side_and_candidate_mappings(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    common = {
        "development_report_paths": fixture["reports"],
        "fixed_threshold_report_paths": fixture["fixed"],
        "license_lock_path": fixture["license"],
        "dataset_readme_path": fixture["readme"],
        "case_count": 18,
    }
    first = build_review_package(
        **common,
        output_root=tmp_path / "first",
        random_seed=b"A" * 32,
    )
    second = build_review_package(
        **common,
        output_root=tmp_path / "second",
        random_seed=b"B" * 32,
    )
    first_vault = _load(first["identityMapping"])
    second_vault = _load(second["identityMapping"])
    assert first["packageId"] != second["packageId"]
    assert [item["caseId"] for item in first_vault["cases"]] != [
        item["caseId"] for item in second_vault["cases"]
    ]
    assert first_vault["ordering"]["seedHex"] == (b"A" * 32).hex()
    assert second_vault["ordering"]["seedHex"] == (b"B" * 32).hex()

    with pytest.raises(BlindSpeakerReviewError, match="256 bits"):
        build_review_package(
            **common,
            output_root=tmp_path / "unsafe",
            random_seed=b"short",
        )


def test_held_out_or_unbalanced_requests_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    report_path = fixture["reports"][0]
    report = _load(report_path)
    report.pop("canonicalSha256")
    report["partition"]["evaluationSplit"] = "held-out"
    report = _canonical(report)
    held_out_path = _write_json(tmp_path / "held-out-report.json", report)

    with pytest.raises(BlindSpeakerReviewError, match="development-only"):
        build_review_package(
            development_report_paths=(held_out_path, fixture["reports"][1]),
            fixed_threshold_report_paths=fixture["fixed"],
            output_root=tmp_path / "held-out-package",
            license_lock_path=fixture["license"],
            dataset_readme_path=fixture["readme"],
            case_count=18,
            random_seed=b"C" * 32,
        )

    with pytest.raises(BlindSpeakerReviewError, match="multiple of 6"):
        build_review_package(
            development_report_paths=fixture["reports"],
            fixed_threshold_report_paths=fixture["fixed"],
            output_root=tmp_path / "bad-count",
            license_lock_path=fixture["license"],
            dataset_readme_path=fixture["readme"],
            case_count=17,
            random_seed=b"D" * 32,
        )
