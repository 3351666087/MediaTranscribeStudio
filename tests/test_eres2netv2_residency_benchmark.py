from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.benchmark_eres2netv2_residency import run_benchmark


class _FakeVerifier:
    instances: list["_FakeVerifier"] = []

    def __init__(self, *, model_path: Path, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._pipeline_instance = object()
        self.calls = 0
        self.released = False
        self.instances.append(self)

    def _embeddings(self, clips: list[object]) -> list[tuple[float, ...]]:
        self.calls += 1
        return [(float(index), 1.0) for index, _ in enumerate(clips)]

    def release_resources(self) -> None:
        self._pipeline_instance = None
        self.released = True


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    manifest = {"schemaVersion": "1.0.0", "model": "fixture"}
    (model / ".mts-model-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(
        b"RIFF"
        + (36 + 32_000).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16_000).to_bytes(4, "little")
        + (32_000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (32_000).to_bytes(4, "little")
        + b"\x00" * 32_000
    )
    return model, audio


def test_residency_benchmark_hashes_inputs_and_reuses_pipeline(
    tmp_path: Path,
) -> None:
    model, audio = _fixture_paths(tmp_path)
    _FakeVerifier.instances.clear()
    snapshots = iter(
        [
            {
                "processRssMb": 100.0,
                "cudaAllocatedMb": 0.0,
                "cudaReservedMb": 0.0,
                "cudaPeakAllocatedMb": 0.0,
                "cudaPeakReservedMb": 0.0,
            },
            {
                "processRssMb": 220.0,
                "cudaAllocatedMb": 180.0,
                "cudaReservedMb": 200.0,
                "cudaPeakAllocatedMb": 190.0,
                "cudaPeakReservedMb": 210.0,
            },
            {
                "processRssMb": 225.0,
                "cudaAllocatedMb": 185.0,
                "cudaReservedMb": 205.0,
                "cudaPeakAllocatedMb": 195.0,
                "cudaPeakReservedMb": 215.0,
            },
            {
                "processRssMb": 140.0,
                "cudaAllocatedMb": 5.0,
                "cudaReservedMb": 10.0,
                "cudaPeakAllocatedMb": 195.0,
                "cudaPeakReservedMb": 215.0,
            },
        ]
    )
    reset_calls: list[bool] = []

    report = run_benchmark(
        model_path=model,
        audio_path=audio,
        device="cpu",
        offsets_ms=[0, 250, 500],
        clip_duration_ms=250,
        warm_runs=2,
        verifier_factory=_FakeVerifier,
        resource_probe=lambda: next(snapshots),
        reset_resource_peaks=lambda: reset_calls.append(True),
    )

    assert report["model"]["manifestFileSha256"] == sha256_file(
        model / ".mts-model-manifest.json"
    )
    assert report["source"]["sha256"] == sha256_file(audio)
    assert report["execution"]["clipCount"] == 3
    assert report["execution"]["embeddingDimensions"] == 2
    assert report["execution"]["residentPipelineReused"] is True
    assert report["execution"]["resourcesReleased"] is True
    assert len(report["execution"]["warmBatchSeconds"]) == 2
    assert report["historicalComparison"][
        "medianWarmPerClipBelowHistorical"
    ] is True
    assert report["schemaVersion"] == "1.1.0"
    assert report["resources"]["peakProcessRssMb"] == 225.0
    assert report["resources"]["peakCudaAllocatedMb"] == 195.0
    assert report["resources"]["peakCudaReservedMb"] == 215.0
    assert report["resources"]["snapshots"]["afterRelease"][
        "cudaAllocatedMb"
    ] == 5.0
    assert reset_calls == [True]
    canonical = dict(report)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)
    assert _FakeVerifier.instances[0].calls == 3
    assert _FakeVerifier.instances[0].released is True


def test_residency_benchmark_rejects_out_of_bounds_clip(
    tmp_path: Path,
) -> None:
    model, audio = _fixture_paths(tmp_path)

    with pytest.raises(ValueError, match="exceeds"):
        run_benchmark(
            model_path=model,
            audio_path=audio,
            device="cpu",
            offsets_ms=[900],
            clip_duration_ms=200,
            warm_runs=1,
            verifier_factory=_FakeVerifier,
        )
