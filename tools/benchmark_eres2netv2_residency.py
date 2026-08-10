"""Benchmark real ERes2NetV2 batch inference and model residency."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file
from backend.production_runners import (
    LocalERes2NetV2Verifier,
    _load_audio,
    _slice_audio,
)

_HISTORICAL_PER_CLIP_SECONDS = 40.0


def _resource_snapshot() -> dict[str, float]:
    snapshot = {
        "processRssMb": 0.0,
        "cudaAllocatedMb": 0.0,
        "cudaReservedMb": 0.0,
        "cudaPeakAllocatedMb": 0.0,
        "cudaPeakReservedMb": 0.0,
    }
    try:
        import psutil

        snapshot["processRssMb"] = psutil.Process().memory_info().rss / (
            1024.0 * 1024.0
        )
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            divisor = 1024.0 * 1024.0
            snapshot.update(
                {
                    "cudaAllocatedMb": (
                        torch.cuda.memory_allocated() / divisor
                    ),
                    "cudaReservedMb": torch.cuda.memory_reserved() / divisor,
                    "cudaPeakAllocatedMb": (
                        torch.cuda.max_memory_allocated() / divisor
                    ),
                    "cudaPeakReservedMb": (
                        torch.cuda.max_memory_reserved() / divisor
                    ),
                }
            )
    except (ImportError, RuntimeError):
        pass
    return snapshot


def _reset_cuda_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def _validated_resource_snapshot(
    probe: Callable[[], Mapping[str, Any]],
) -> dict[str, float]:
    required = {
        "processRssMb",
        "cudaAllocatedMb",
        "cudaReservedMb",
        "cudaPeakAllocatedMb",
        "cudaPeakReservedMb",
    }
    raw = probe()
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise RuntimeError("resource probe returned an invalid snapshot")
    snapshot = {key: float(raw[key]) for key in sorted(required)}
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in snapshot.values()
    ):
        raise RuntimeError("resource probe returned an invalid value")
    return snapshot


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _model_manifest_evidence(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / ".mts-model-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "ERes2NetV2 model is missing .mts-model-manifest.json"
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("ERes2NetV2 model manifest must be an object")
    return {
        "path": str(model_path),
        "manifestPath": str(manifest_path),
        "manifestFileSha256": sha256_file(manifest_path),
        "manifestCanonicalSha256": canonical_json_sha256(document),
    }


def run_benchmark(
    *,
    model_path: Path,
    audio_path: Path,
    device: str,
    offsets_ms: Sequence[int],
    clip_duration_ms: int,
    warm_runs: int,
    verifier_factory: Callable[..., Any] = LocalERes2NetV2Verifier,
    resource_probe: Callable[[], Mapping[str, Any]] = _resource_snapshot,
    reset_resource_peaks: Callable[[], None] = _reset_cuda_peaks,
) -> dict[str, Any]:
    resolved_model = model_path.resolve(strict=True)
    resolved_audio = audio_path.resolve(strict=True)
    if not resolved_model.is_dir():
        raise ValueError("ERes2NetV2 model path must be a directory")
    if not resolved_audio.is_file():
        raise ValueError("benchmark audio path must be a file")
    if not offsets_ms:
        raise ValueError("at least one clip offset is required")
    if clip_duration_ms < 1 or warm_runs < 1:
        raise ValueError("clip duration and warm run count must be positive")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
        for offset in offsets_ms
    ):
        raise ValueError("clip offsets must be non-negative integers")

    samples, sample_rate = _load_audio(resolved_audio)
    audio_duration_ms = round(len(samples) * 1000.0 / sample_rate)
    clip_ends_ms = [
        offset + clip_duration_ms for offset in offsets_ms
    ]
    if any(end > audio_duration_ms for end in clip_ends_ms):
        raise ValueError("benchmark clip exceeds the source audio duration")
    clips = [
        _slice_audio(
            samples,
            sample_rate,
            offset,
            offset + clip_duration_ms,
        )
        for offset in offsets_ms
    ]

    reset_resource_peaks()
    resource_before = _validated_resource_snapshot(resource_probe)
    verifier = verifier_factory(
        model_path=resolved_model,
        device=device,
    )
    released = False
    try:
        started = time.perf_counter()
        cold_vectors = verifier._embeddings(clips)
        cold_seconds = time.perf_counter() - started
        resource_after_cold = _validated_resource_snapshot(resource_probe)
        pipeline_instance = verifier._pipeline_instance
        if pipeline_instance is None:
            raise RuntimeError("ERes2NetV2 pipeline did not remain resident")
        pipeline_identity = id(pipeline_instance)
        del pipeline_instance
        dimensions = {
            len(vector) for vector in cold_vectors
        }
        if len(cold_vectors) != len(clips) or len(dimensions) != 1:
            raise RuntimeError(
                "ERes2NetV2 returned an invalid cold batch"
            )

        warm_seconds: list[float] = []
        residency_stable = True
        for _ in range(warm_runs):
            started = time.perf_counter()
            vectors = verifier._embeddings(clips)
            warm_seconds.append(time.perf_counter() - started)
            residency_stable = (
                residency_stable
                and verifier._pipeline_instance is not None
                and id(verifier._pipeline_instance) == pipeline_identity
            )
            if (
                len(vectors) != len(clips)
                or {len(vector) for vector in vectors} != dimensions
            ):
                raise RuntimeError(
                    "ERes2NetV2 returned an inconsistent warm batch"
                )
        resource_after_warm = _validated_resource_snapshot(resource_probe)
    finally:
        verifier.release_resources()
        released = verifier._pipeline_instance is None
    resource_after_release = _validated_resource_snapshot(resource_probe)

    clip_count = len(clips)
    median_warm_seconds = statistics.median(warm_seconds)
    report: dict[str, Any] = {
        "schemaVersion": "1.1.0",
        "benchmark": "eres2netv2-batch-residency",
        "model": _model_manifest_evidence(resolved_model),
        "source": {
            "path": str(resolved_audio),
            "sha256": sha256_file(resolved_audio),
            "durationMs": audio_duration_ms,
            "sampleRateHz": sample_rate,
        },
        "execution": {
            "device": device,
            "clipOffsetsMs": list(offsets_ms),
            "clipDurationMs": clip_duration_ms,
            "clipCount": clip_count,
            "warmRunCount": warm_runs,
            "embeddingDimensions": next(iter(dimensions)),
            "coldBatchSeconds": cold_seconds,
            "coldPerClipSeconds": cold_seconds / clip_count,
            "warmBatchSeconds": warm_seconds,
            "medianWarmBatchSeconds": median_warm_seconds,
            "medianWarmPerClipSeconds": (
                median_warm_seconds / clip_count
            ),
            "residentPipelineReused": residency_stable,
            "resourcesReleased": released,
        },
        "historicalComparison": {
            "historicalPerClipSeconds": _HISTORICAL_PER_CLIP_SECONDS,
            "coldPerClipBelowHistorical": (
                cold_seconds / clip_count
                < _HISTORICAL_PER_CLIP_SECONDS
            ),
            "medianWarmPerClipBelowHistorical": (
                median_warm_seconds / clip_count
                < _HISTORICAL_PER_CLIP_SECONDS
            ),
        },
        "resources": {
            "snapshots": {
                "beforeModelLoad": resource_before,
                "afterColdBatch": resource_after_cold,
                "afterWarmBatches": resource_after_warm,
                "afterRelease": resource_after_release,
            },
            "peakProcessRssMb": max(
                item["processRssMb"]
                for item in (
                    resource_before,
                    resource_after_cold,
                    resource_after_warm,
                    resource_after_release,
                )
            ),
            "peakCudaAllocatedMb": max(
                item["cudaPeakAllocatedMb"]
                for item in (
                    resource_before,
                    resource_after_cold,
                    resource_after_warm,
                    resource_after_release,
                )
            ),
            "peakCudaReservedMb": max(
                item["cudaPeakReservedMb"]
                for item in (
                    resource_before,
                    resource_after_cold,
                    resource_after_warm,
                    resource_after_release,
                )
            ),
        },
    }
    numeric_values = (
        cold_seconds,
        *warm_seconds,
        cold_seconds / clip_count,
        median_warm_seconds / clip_count,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric_values):
        raise RuntimeError("benchmark produced invalid timing values")
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark report: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--offset-ms",
        type=_nonnegative_integer,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--clip-duration-ms",
        type=_positive_integer,
        default=2_500,
    )
    parser.add_argument(
        "--warm-runs",
        type=_positive_integer,
        default=3,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        model_path=args.model_path,
        audio_path=args.audio_path,
        device=args.device,
        offsets_ms=args.offset_ms,
        clip_duration_ms=args.clip_duration_ms,
        warm_runs=args.warm_runs,
    )
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
