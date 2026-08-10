from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.benchmark_redimnet2_verification import load_trial_manifest
from tools.build_cnceleb_speaker_trials import (
    CNCelebTrialError,
    build_gate,
    parse_clip_identity,
)
from tools.evaluate_eres2netv2_trials import _load_trial_manifest


def _archive_evidence(root: Path) -> Path:
    archive = root / "cn-celeb_v2.tar.gz"
    archive.write_bytes(b"archive-fixture")
    evidence = {
        "schemaVersion": "1.0.0",
        "datasetKey": "openslr-cnceleb1-v2",
        "validation": {
            "archivePath": str(archive),
            "archiveBytes": archive.stat().st_size,
            "archiveSha256": sha256_file(archive),
            "completeTarScanPassed": True,
            "safeMemberPolicyPassed": True,
        },
    }
    evidence["canonicalSha256"] = canonical_json_sha256(evidence)
    path = root / "archive-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _dataset(root: Path, *, speaker_count: int = 6) -> Path:
    dataset = root / "CN-Celeb_flac"
    test_root = dataset / "eval" / "test"
    test_root.mkdir(parents=True)
    for speaker_number in range(800, 800 + speaker_count):
        speaker = f"id{speaker_number:05d}"
        for recording in (1, 2, 3):
            path = test_root / (
                f"{speaker}-live_broadcast-{recording:02d}-001.flac"
            )
            path.write_bytes(f"{speaker}:{recording}".encode("ascii"))
    return dataset


def _pcm_loader(path: Path, frames: int, sample_rate: int) -> bytes:
    assert sample_rate == 16_000
    value = int(path.stem[-3:]) + 1
    return value.to_bytes(2, "little", signed=True) * frames


def test_clip_identity_preserves_genre_and_original_recording() -> None:
    parsed = parse_clip_identity(
        Path("id00915-live_broadcast-07-014.flac")
    )

    assert parsed == {
        "speaker": "id00915",
        "genre": "live_broadcast",
        "recording": "07",
        "segment": "014",
        "originalRecordingId": "id00915-live_broadcast-07",
    }
    with pytest.raises(CNCelebTrialError, match="unsupported"):
        parse_clip_identity(Path("id00915-live_broadcast-07.wav"))


def test_gate_is_speaker_disjoint_cross_recording_and_evaluator_compatible(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    result = build_gate(
        dataset_root=dataset,
        archive_evidence_path=_archive_evidence(tmp_path),
        output_directory=tmp_path / "gate",
        speakers_per_split=2,
        clips_per_speaker=2,
        clip_duration_ms=100,
        padding_ms=10,
        maximum_trials_per_class=8,
        random_seed=17,
        pcm_loader=_pcm_loader,
    )

    assert set(result["developmentSpeakers"]).isdisjoint(
        result["heldOutSpeakers"]
    )
    for split in ("development", "held-out"):
        partition = result["partitions"][split]
        manifest = partition["manifest"]
        clips = {clip["clipId"]: clip for clip in manifest["clips"]}
        assert manifest["counts"] == {
            "speakers": 2,
            "clips": 4,
            "clipsPerSpeaker": 2,
            "sameSpeakerTrials": 2,
            "differentSpeakerTrials": 2,
            "totalTrials": 4,
        }
        for trial in manifest["trials"]:
            left = clips[trial["enrollmentClipId"]]
            right = clips[trial["testClipId"]]
            assert left["originalRecordingId"] != right["originalRecordingId"]
            if trial["sameSpeaker"]:
                assert left["speakerId"] == right["speakerId"]
        body = dict(manifest)
        declared = body.pop("canonicalSha256")
        assert declared == canonical_json_sha256(body)
        with wave.open(str(partition["audioPath"]), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16_000

        shared = _load_trial_manifest(partition["manifestPath"])
        redimnet = load_trial_manifest(partition["manifestPath"])
        assert shared["evaluationSplit"] == split
        assert redimnet["counts"] == manifest["counts"]


def test_gate_rejects_speakers_without_distinct_original_recordings(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "CN-Celeb_flac"
    test_root = dataset / "eval" / "test"
    test_root.mkdir(parents=True)
    for speaker_number in range(800, 804):
        speaker = f"id{speaker_number:05d}"
        for segment in (1, 2):
            (test_root / f"{speaker}-speech-01-{segment:03d}.flac").write_bytes(
                b"fixture"
            )

    with pytest.raises(CNCelebTrialError, match="distinct readable"):
        build_gate(
            dataset_root=dataset,
            archive_evidence_path=_archive_evidence(tmp_path),
            output_directory=tmp_path / "gate",
            speakers_per_split=2,
            clips_per_speaker=2,
            clip_duration_ms=100,
            pcm_loader=_pcm_loader,
        )
