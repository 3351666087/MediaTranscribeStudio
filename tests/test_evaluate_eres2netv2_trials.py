from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.evaluate_eres2netv2_trials import (
    SpeakerVerificationEvaluationError,
    _embedding_set_sha256,
    run_evaluation,
)


class _FakeVerifier:
    def __init__(self, *, model_path: Path, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline_instance = object()
        self.offset = 0

    def _embeddings(self, clips: list[object]) -> list[tuple[float, ...]]:
        vectors = (
            (1.0, 0.0),
            (0.98, 0.02),
            (0.0, 1.0),
            (0.02, 0.98),
        )
        output = list(vectors[self.offset : self.offset + len(clips)])
        self.offset += len(clips)
        return output

    def release_resources(self) -> None:
        self._pipeline_instance = None


def _resource(value: float) -> dict[str, float]:
    return {
        "processRssMb": value,
        "cudaAllocatedMb": value,
        "cudaReservedMb": value,
        "cudaPeakAllocatedMb": value,
        "cudaPeakReservedMb": value,
    }


def _write_wave(path: Path) -> None:
    samples = np.zeros(16_000 * 16, dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.tobytes())


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "source.wav"
    _write_wave(audio)
    annotation = tmp_path / "source.json"
    annotation.write_text(
        json.dumps(
            {
                "timestamps_start": [0.0, 8.0, 4.0, 12.0],
                "timestamps_end": [3.0, 11.0, 7.0, 15.0],
                "speakers": ["speaker-a", "speaker-a", "speaker-b", "speaker-b"],
            }
        ),
        encoding="utf-8",
    )
    clips = [
        {
            "clipId": "a-1",
            "speakerId": "speaker-a",
            "startMs": 0,
            "endMs": 2500,
            "audio": str(audio),
            "sourceRecordingId": "recording-1",
            "evaluationSplit": "development",
        },
        {
            "clipId": "a-2",
            "speakerId": "speaker-a",
            "startMs": 8000,
            "endMs": 10500,
            "audio": str(audio),
            "sourceRecordingId": "recording-1",
            "evaluationSplit": "development",
        },
        {
            "clipId": "b-1",
            "speakerId": "speaker-b",
            "startMs": 4000,
            "endMs": 6500,
            "audio": str(audio),
            "sourceRecordingId": "recording-1",
            "evaluationSplit": "development",
        },
        {
            "clipId": "b-2",
            "speakerId": "speaker-b",
            "startMs": 12000,
            "endMs": 14500,
            "audio": str(audio),
            "sourceRecordingId": "recording-1",
            "evaluationSplit": "development",
        },
    ]
    trials = [
        {
            "trialId": "same-a",
            "enrollmentClipId": "a-1",
            "testClipId": "a-2",
            "sameSpeaker": True,
            "evaluationSplit": "development",
        },
        {
            "trialId": "same-b",
            "enrollmentClipId": "b-1",
            "testClipId": "b-2",
            "sameSpeaker": True,
            "evaluationSplit": "development",
        },
        {
            "trialId": "different-1",
            "enrollmentClipId": "a-1",
            "testClipId": "b-1",
            "sameSpeaker": False,
            "evaluationSplit": "development",
        },
        {
            "trialId": "different-2",
            "enrollmentClipId": "a-2",
            "testClipId": "b-2",
            "sameSpeaker": False,
            "evaluationSplit": "development",
        },
    ]
    document = {
        "schemaVersion": "1.0.0",
        "libraryId": "fixture-speaker-verification-v1",
        "randomSeed": 1,
        "selection": {},
        "source": {
            "audioPath": str(audio),
            "audioBytes": audio.stat().st_size,
            "audioSha256": sha256_file(audio),
            "annotationPath": str(annotation),
            "annotationBytes": annotation.stat().st_size,
            "annotationSha256": sha256_file(annotation),
            "recordingId": "recording-1",
            "dataset": "fixture",
            "revision": "frozen",
            "audio": {
                "sampleRateHz": 16_000,
                "durationMs": 16_000,
            },
        },
        "clips": clips,
        "trials": trials,
        "counts": {
            "clips": 4,
            "clipsPerSpeaker": 2,
            "speakers": 2,
            "sameSpeakerTrials": 2,
            "differentSpeakerTrials": 2,
            "totalTrials": 4,
        },
    }
    document["canonicalSha256"] = canonical_json_sha256(document)
    trial_manifest = tmp_path / "trials.json"
    trial_manifest.write_text(json.dumps(document), encoding="utf-8")

    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weight.bin"
    weight.write_bytes(b"weight")
    model_manifest = {
        "schemaVersion": "1.0.0",
        "modelKey": "fixture-eres2net",
        "provider": "fixture",
        "repoId": "fixture/eres2net",
        "revision": "v1",
        "lockSha256": "a" * 64,
        "totalBytes": weight.stat().st_size,
        "files": [
            {
                "path": weight.name,
                "size": weight.stat().st_size,
                "sha256": sha256_file(weight),
            }
        ],
    }
    (model / ".mts-model-manifest.json").write_text(
        json.dumps(model_manifest), encoding="utf-8"
    )
    return model, trial_manifest


def test_frozen_trial_evaluation_scores_and_binds_all_evidence(
    tmp_path: Path,
) -> None:
    model, manifest = _fixture(tmp_path)
    resources = iter((_resource(1.0), _resource(5.0), _resource(2.0)))
    resets: list[bool] = []

    report = run_evaluation(
        model_path=model,
        trial_manifest_path=manifest,
        device="cpu",
        batch_size=2,
        verifier_factory=_FakeVerifier,
        resource_probe=lambda: next(resources),
        reset_resource_peaks=lambda: resets.append(True),
    )

    assert report["scores"]["rocAuc"] == 1.0
    assert report["scores"]["equalErrorRate"] == 0.0
    assert report["partition"]["recordingCount"] == 1
    assert report["partition"]["evaluationSplit"] == "development"
    assert report["promotionPolicy"]["promotionEligibleBySplit"] is False
    assert report["model"]["integrityVerified"] is True
    assert report["resources"]["peakProcessRssMb"] == 5.0
    assert report["execution"]["adapterResourcesReleased"] is True
    assert report["execution"]["embeddingSetHashAlgorithm"] == (
        "clip-id-sorted-float32-le-v1"
    )
    assert len(report["execution"]["embeddingSetSha256"]) == 64
    assert resets == [True]
    canonical = dict(report)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)


def test_embedding_set_digest_is_order_independent_and_value_bound() -> None:
    first = {"b": (0.0, 1.0), "a": (1.0, 0.0)}
    reordered = {"a": first["a"], "b": first["b"]}
    changed = {"a": first["a"], "b": (0.0, 0.9)}

    assert _embedding_set_sha256(first) == _embedding_set_sha256(reordered)
    assert _embedding_set_sha256(first) != _embedding_set_sha256(changed)


def test_trial_manifest_canonical_tampering_fails_closed(tmp_path: Path) -> None:
    model, manifest = _fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["clips"][0]["speakerId"] = "speaker-b"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        SpeakerVerificationEvaluationError,
        match="canonicalSha256",
    ):
        run_evaluation(
            model_path=model,
            trial_manifest_path=manifest,
            device="cpu",
            batch_size=2,
            verifier_factory=_FakeVerifier,
        )
