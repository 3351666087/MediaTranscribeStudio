"""Validate ReDimNet2 batch consistency and explicit CUDA release."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from tools.benchmark_redimnet2_verification import (  # noqa: E402
    EXCLUDED_CLASSIFICATION_KEYS,
    ReDimNet2BenchmarkError,
    _load_model,
    load_trial_manifest,
    verification_metrics,
    verify_model_snapshot,
    verify_wespeaker_source,
)


DEFAULT_BATCH_SIZES = (4, 8, 12, 16)
CONSISTENCY_ABSOLUTE_TOLERANCE = 1e-5
CONSISTENCY_RELATIVE_TOLERANCE = 1e-5
CUDA_RELEASE_TOLERANCE_BYTES = 16 * 1024 * 1024
_RESOURCE_KEYS = frozenset(
    {
        "processRssBytes",
        "cudaAllocatedBytes",
        "cudaReservedBytes",
        "cudaPeakAllocatedBytes",
        "cudaPeakReservedBytes",
    }
)


def _batch_sizes(value: str) -> tuple[int, ...]:
    raw_parts = value.split(",")
    if not raw_parts or any(not part.strip() for part in raw_parts):
        raise argparse.ArgumentTypeError(
            "batch sizes must be a comma-separated list of positive integers"
        )
    try:
        parsed = tuple(int(part.strip()) for part in raw_parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch sizes must be a comma-separated list of positive integers"
        ) from exc
    if any(size < 1 for size in parsed):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return parsed


def _resource_snapshot(torch_module: Any, target_device: Any) -> dict[str, int]:
    try:
        import psutil

        process_rss = int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError) as exc:
        raise ReDimNet2BenchmarkError(
            "psutil is required for Windows process RSS evidence"
        ) from exc

    snapshot = {
        "processRssBytes": process_rss,
        "cudaAllocatedBytes": 0,
        "cudaReservedBytes": 0,
        "cudaPeakAllocatedBytes": 0,
        "cudaPeakReservedBytes": 0,
    }
    if target_device.type == "cuda":
        torch_module.cuda.synchronize(target_device)
        snapshot.update(
            {
                "cudaAllocatedBytes": int(
                    torch_module.cuda.memory_allocated(target_device)
                ),
                "cudaReservedBytes": int(
                    torch_module.cuda.memory_reserved(target_device)
                ),
                "cudaPeakAllocatedBytes": int(
                    torch_module.cuda.max_memory_allocated(target_device)
                ),
                "cudaPeakReservedBytes": int(
                    torch_module.cuda.max_memory_reserved(target_device)
                ),
            }
        )
    return snapshot


def _validated_resource_snapshot(
    probe: Callable[[], Mapping[str, Any]],
) -> dict[str, int]:
    raw = probe()
    if not isinstance(raw, Mapping) or set(raw) != _RESOURCE_KEYS:
        raise ReDimNet2BenchmarkError(
            "resource probe returned an invalid snapshot"
        )
    snapshot: dict[str, int] = {}
    for key in sorted(_RESOURCE_KEYS):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReDimNet2BenchmarkError(
                f"resource probe returned an invalid value for {key}"
            )
        snapshot[key] = value
    return snapshot


def _reset_cuda_peaks(torch_module: Any, target_device: Any) -> None:
    if target_device.type == "cuda":
        torch_module.cuda.synchronize(target_device)
        torch_module.cuda.reset_peak_memory_stats(target_device)


def _empty_cuda_cache(
    torch_module: Any, target_device: Any
) -> dict[str, bool]:
    evidence = {
        "cudaCacheEmptied": False,
        "cublasWorkspacesCleared": False,
        "cufftPlanCacheCleared": False,
    }
    if target_device.type != "cuda":
        return evidence
    torch_module.cuda.synchronize(target_device)
    clear_cublas = getattr(
        getattr(torch_module, "_C", object()),
        "_cuda_clearCublasWorkspaces",
        None,
    )
    if callable(clear_cublas):
        clear_cublas()
        evidence["cublasWorkspacesCleared"] = True
    cufft_cache = getattr(torch_module.backends.cuda, "cufft_plan_cache", None)
    if cufft_cache is not None:
        cufft_cache.clear()
        evidence["cufftPlanCacheCleared"] = True
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize(target_device)
    evidence["cudaCacheEmptied"] = True
    return evidence


def _configure_deterministic_inference(torch_module: Any) -> dict[str, Any]:
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    evidence = {
        "deterministicAlgorithmsEnabled": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministicAlgorithmsWarnOnly": bool(
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnnBenchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnnDeterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnnAllowTf32": bool(torch_module.backends.cudnn.allow_tf32),
        "cudaMatmulAllowTf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "float32MatmulPrecision": torch_module.get_float32_matmul_precision(),
        "cublasWorkspaceConfig": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if (
        not evidence["deterministicAlgorithmsEnabled"]
        or evidence["deterministicAlgorithmsWarnOnly"]
        or evidence["cudnnBenchmark"]
        or not evidence["cudnnDeterministic"]
        or evidence["cudnnAllowTf32"]
        or evidence["cudaMatmulAllowTf32"]
        or evidence["cublasWorkspaceConfig"] != ":4096:8"
    ):
        raise ReDimNet2BenchmarkError(
            "deterministic inference configuration did not take effect"
        )
    return evidence


def _release_model_holder(
    holder: MutableMapping[str, Any],
    *,
    torch_module: Any,
    target_device: Any,
    collect_garbage: Callable[[], int] = gc.collect,
) -> dict[str, Any]:
    if set(holder) != {"model"}:
        raise ReDimNet2BenchmarkError(
            "model holder must contain exactly one model reference"
        )
    model = holder.pop("model")
    try:
        model_reference = weakref.ref(model)
    except TypeError as exc:
        raise ReDimNet2BenchmarkError(
            "ReDimNet2 model does not support release tracking"
        ) from exc
    model_identity = id(model)
    del model
    first_gc_count = int(collect_garbage())
    first_cleanup = _empty_cuda_cache(torch_module, target_device)
    second_gc_count = int(collect_garbage())
    second_cleanup = _empty_cuda_cache(torch_module, target_device)
    return {
        "modelObjectIdentity": model_identity,
        "modelObjectCollected": model_reference() is None,
        "garbageCollectedObjectCount": first_gc_count + second_gc_count,
        "cudaCacheEmptied": (
            first_cleanup["cudaCacheEmptied"]
            and second_cleanup["cudaCacheEmptied"]
        ),
        "cublasWorkspacesCleared": (
            first_cleanup["cublasWorkspacesCleared"]
            and second_cleanup["cublasWorkspacesCleared"]
        ),
        "cufftPlanCacheCleared": (
            first_cleanup["cufftPlanCacheCleared"]
            and second_cleanup["cufftPlanCacheCleared"]
        ),
    }


def _load_clips(
    manifest: Mapping[str, Any],
    *,
    numpy_module: Any,
    soundfile_module: Any,
) -> tuple[list[Mapping[str, Any]], list[Any], int]:
    source = manifest["source"]
    audio, sample_rate = soundfile_module.read(
        Path(str(source["audioPath"])),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != source["audio"]["sampleRateHz"] or audio.shape[1] != 1:
        raise ReDimNet2BenchmarkError(
            "batch scan currently requires the frozen mono source audio"
        )
    clip_rows = sorted(manifest["clips"], key=lambda row: str(row["clipId"]))
    clip_waves: list[Any] = []
    expected_samples: int | None = None
    for clip in clip_rows:
        start_sample = round(int(clip["startMs"]) * sample_rate / 1000.0)
        end_sample = round(int(clip["endMs"]) * sample_rate / 1000.0)
        wave = numpy_module.asarray(
            audio[start_sample:end_sample, 0], dtype=numpy_module.float32
        )
        if wave.size < 1:
            raise ReDimNet2BenchmarkError("trial clip is empty")
        if expected_samples is None:
            expected_samples = int(wave.size)
        elif wave.size != expected_samples:
            raise ReDimNet2BenchmarkError(
                "trial clips do not have equal duration"
            )
        clip_waves.append(wave)
    return clip_rows, clip_waves, int(sample_rate)


def _vector_set_sha256(
    vectors: Mapping[str, Any], *, numpy_module: Any
) -> str:
    digest = hashlib.sha256()
    for clip_id in sorted(vectors):
        encoded_id = clip_id.encode("utf-8")
        vector = numpy_module.ascontiguousarray(
            numpy_module.asarray(vectors[clip_id], dtype="<f4")
        )
        if vector.ndim != 1 or vector.size < 1:
            raise ReDimNet2BenchmarkError("embedding vector is invalid")
        digest.update(len(encoded_id).to_bytes(4, "big"))
        digest.update(encoded_id)
        digest.update(int(vector.size).to_bytes(8, "big"))
        digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


def _trial_scores(
    manifest: Mapping[str, Any],
    vectors: Mapping[str, Any],
    *,
    numpy_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    metric_input: list[tuple[float, bool]] = []
    for trial in manifest["trials"]:
        left_id = str(trial["enrollmentClipId"])
        right_id = str(trial["testClipId"])
        try:
            left = vectors[left_id]
            right = vectors[right_id]
        except KeyError as exc:
            raise ReDimNet2BenchmarkError(
                "embedding set does not cover every frozen trial"
            ) from exc
        score = float(numpy_module.dot(left, right))
        if not math.isfinite(score):
            raise ReDimNet2BenchmarkError("trial score is non-finite")
        same_speaker = bool(trial["sameSpeaker"])
        metric_input.append((score, same_speaker))
        rows.append(
            {
                "trialId": str(trial["trialId"]),
                "enrollmentClipId": left_id,
                "testClipId": right_id,
                "sameSpeaker": same_speaker,
                "cosine": score,
            }
        )
    return rows, verification_metrics(metric_input)


def _numeric_consistency(
    reference: Any,
    candidate: Any,
    *,
    numpy_module: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    left = numpy_module.asarray(reference, dtype=numpy_module.float64)
    right = numpy_module.asarray(candidate, dtype=numpy_module.float64)
    if left.shape != right.shape:
        raise ReDimNet2BenchmarkError("consistency arrays have different shapes")
    if left.size < 1:
        raise ReDimNet2BenchmarkError("consistency arrays are empty")
    if not bool(numpy_module.isfinite(left).all()) or not bool(
        numpy_module.isfinite(right).all()
    ):
        raise ReDimNet2BenchmarkError("consistency arrays contain non-finite values")
    absolute_delta = numpy_module.abs(right - left)
    allowed_delta = (
        absolute_tolerance + relative_tolerance * numpy_module.abs(left)
    )
    relative_delta = absolute_delta / numpy_module.maximum(
        numpy_module.abs(left), numpy_module.finfo(numpy_module.float64).tiny
    )
    tolerance_ratio = absolute_delta / allowed_delta
    return {
        "consistent": bool((absolute_delta <= allowed_delta).all()),
        "valueCount": int(left.size),
        "maxAbsoluteDelta": float(absolute_delta.max()),
        "maxRelativeDelta": float(relative_delta.max()),
        "maxToleranceRatio": float(tolerance_ratio.max()),
    }


def _compare_vector_sets(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    numpy_module: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ReDimNet2BenchmarkError("embedding sets cover different clips")
    ordered_ids = sorted(reference)
    left = numpy_module.concatenate(
        [numpy_module.asarray(reference[clip_id]).reshape(-1) for clip_id in ordered_ids]
    )
    right = numpy_module.concatenate(
        [numpy_module.asarray(candidate[clip_id]).reshape(-1) for clip_id in ordered_ids]
    )
    comparison = _numeric_consistency(
        left,
        right,
        numpy_module=numpy_module,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    comparison["clipCount"] = len(ordered_ids)
    return comparison


def _compare_trial_score_sets(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    numpy_module: Any,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    metadata_keys = (
        "trialId",
        "enrollmentClipId",
        "testClipId",
        "sameSpeaker",
    )
    reference_by_id = {str(row["trialId"]): row for row in reference}
    candidate_by_id = {str(row["trialId"]): row for row in candidate}
    if (
        len(reference_by_id) != len(reference)
        or len(candidate_by_id) != len(candidate)
        or set(reference_by_id) != set(candidate_by_id)
    ):
        raise ReDimNet2BenchmarkError("trial score sets cover different trials")
    ordered_ids = sorted(reference_by_id)
    for trial_id in ordered_ids:
        left_row = reference_by_id[trial_id]
        right_row = candidate_by_id[trial_id]
        if any(left_row[key] != right_row[key] for key in metadata_keys):
            raise ReDimNet2BenchmarkError("trial score metadata mismatches")
    comparison = _numeric_consistency(
        [reference_by_id[trial_id]["cosine"] for trial_id in ordered_ids],
        [candidate_by_id[trial_id]["cosine"] for trial_id in ordered_ids],
        numpy_module=numpy_module,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    comparison["trialCount"] = len(ordered_ids)
    return comparison


def _run_embedding_scan(
    *,
    model: Any,
    clip_rows: Sequence[Mapping[str, Any]],
    clip_waves: Sequence[Any],
    batch_size: int,
    torch_module: Any,
    numpy_module: Any,
    target_device: Any,
) -> tuple[dict[str, Any], float]:
    vectors: dict[str, Any] = {}
    started = time.perf_counter()
    with torch_module.inference_mode():
        for offset in range(0, len(clip_rows), batch_size):
            rows = clip_rows[offset : offset + batch_size]
            batch = torch_module.from_numpy(
                numpy_module.stack(clip_waves[offset : offset + batch_size])
            ).to(target_device)
            lengths = torch_module.full(
                (batch.shape[0],),
                batch.shape[1],
                dtype=torch_module.long,
                device=target_device,
            )
            features, _ = model.frontend(batch, lengths)
            embeddings = torch_module.nn.functional.normalize(
                model(features).float(), dim=-1
            ).detach().cpu().numpy()
            if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
                raise ReDimNet2BenchmarkError(
                    "ReDimNet2 returned an invalid embedding batch"
                )
            for row, vector in zip(rows, embeddings):
                clip_id = str(row["clipId"])
                vectors[clip_id] = numpy_module.asarray(
                    vector, dtype=numpy_module.float32
                ).copy()
            del embeddings, features, lengths, batch
    if target_device.type == "cuda":
        torch_module.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - started
    if set(vectors) != {str(row["clipId"]) for row in clip_rows}:
        raise ReDimNet2BenchmarkError(
            "ReDimNet2 did not return every frozen clip embedding"
        )
    return vectors, elapsed


def _attach_canonical_digest(report: Mapping[str, Any]) -> dict[str, Any]:
    if "canonicalSha256" in report:
        raise ReDimNet2BenchmarkError(
            "report already contains a canonical digest"
        )
    finalized = dict(report)
    finalized["canonicalSha256"] = canonical_json_sha256(finalized)
    return finalized


def run_batch_scan(
    *,
    model_path: Path,
    wespeaker_source: Path,
    trial_manifest_path: Path,
    device: str,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    absolute_tolerance: float = CONSISTENCY_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = CONSISTENCY_RELATIVE_TOLERANCE,
    model_loader: Callable[[Path, Path, str], Any] = _load_model,
    resource_probe_factory: Callable[[Any, Any], Mapping[str, Any]] = (
        _resource_snapshot
    ),
) -> dict[str, Any]:
    normalized_batch_sizes = tuple(batch_sizes)
    if (
        not normalized_batch_sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in normalized_batch_sizes
        )
        or len(set(normalized_batch_sizes)) != len(normalized_batch_sizes)
    ):
        raise ReDimNet2BenchmarkError(
            "batch sizes must be unique positive integers"
        )
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance <= 0.0
        or relative_tolerance < 0.0
    ):
        raise ReDimNet2BenchmarkError("consistency tolerances are invalid")

    model_evidence = verify_model_snapshot(model_path)
    source_evidence = verify_wespeaker_source(wespeaker_source)
    manifest = load_trial_manifest(trial_manifest_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise ReDimNet2BenchmarkError(
            "NumPy, SoundFile, PyTorch, and psutil are required"
        ) from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ReDimNet2BenchmarkError("CUDA benchmark requested but unavailable")
    determinism_evidence = _configure_deterministic_inference(torch)

    clip_rows, clip_waves, sample_rate = _load_clips(
        manifest,
        numpy_module=np,
        soundfile_module=sf,
    )
    resource_probe = lambda: resource_probe_factory(torch, target_device)
    gc.collect()
    _empty_cuda_cache(torch, target_device)
    _reset_cuda_peaks(torch, target_device)
    resources_before_load = _validated_resource_snapshot(resource_probe)

    _reset_cuda_peaks(torch, target_device)
    load_started = time.perf_counter()
    model_holder: dict[str, Any] = {
        "model": model_loader(
            model_path.resolve(), wespeaker_source.resolve(), device
        )
    }
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    model_load_seconds = time.perf_counter() - load_started
    resources_after_load = _validated_resource_snapshot(resource_probe)

    reference_vectors: dict[str, Any] | None = None
    reference_scores: list[dict[str, Any]] | None = None
    scans: list[dict[str, Any]] = []
    all_vectors_consistent = True
    all_scores_consistent = True
    try:
        for batch_size in normalized_batch_sizes:
            _reset_cuda_peaks(torch, target_device)
            resources_before_scan = _validated_resource_snapshot(resource_probe)
            vectors, embedding_seconds = _run_embedding_scan(
                model=model_holder["model"],
                clip_rows=clip_rows,
                clip_waves=clip_waves,
                batch_size=batch_size,
                torch_module=torch,
                numpy_module=np,
                target_device=target_device,
            )
            scoring_started = time.perf_counter()
            trial_scores, metrics = _trial_scores(
                manifest, vectors, numpy_module=np
            )
            scoring_seconds = time.perf_counter() - scoring_started
            resources_after_scan = _validated_resource_snapshot(resource_probe)
            vector_digest = _vector_set_sha256(vectors, numpy_module=np)
            score_digest = canonical_json_sha256(trial_scores)

            if reference_vectors is None or reference_scores is None:
                vector_consistency = {
                    "reference": True,
                    "consistent": True,
                    "clipCount": len(clip_rows),
                    "valueCount": sum(
                        int(np.asarray(vector).size)
                        for vector in vectors.values()
                    ),
                    "maxAbsoluteDelta": 0.0,
                    "maxRelativeDelta": 0.0,
                    "maxToleranceRatio": 0.0,
                }
                score_consistency = {
                    "reference": True,
                    "consistent": True,
                    "trialCount": len(trial_scores),
                    "valueCount": len(trial_scores),
                    "maxAbsoluteDelta": 0.0,
                    "maxRelativeDelta": 0.0,
                    "maxToleranceRatio": 0.0,
                }
                reference_vectors = vectors
                reference_scores = trial_scores
            else:
                vector_consistency = _compare_vector_sets(
                    reference_vectors,
                    vectors,
                    numpy_module=np,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                vector_consistency["reference"] = False
                score_consistency = _compare_trial_score_sets(
                    reference_scores,
                    trial_scores,
                    numpy_module=np,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                score_consistency["reference"] = False
                all_vectors_consistent = (
                    all_vectors_consistent
                    and bool(vector_consistency["consistent"])
                )
                all_scores_consistent = (
                    all_scores_consistent
                    and bool(score_consistency["consistent"])
                )

            scans.append(
                {
                    "batchSize": batch_size,
                    "embeddingSeconds": embedding_seconds,
                    "embeddingSecondsPerClip": (
                        embedding_seconds / len(clip_rows)
                    ),
                    "scoringSeconds": scoring_seconds,
                    "totalScanSeconds": embedding_seconds + scoring_seconds,
                    "embeddingDimensions": int(
                        next(iter(vectors.values())).shape[-1]
                    ),
                    "vectorSetSha256": vector_digest,
                    "trialScoreSetCanonicalSha256": score_digest,
                    "metrics": metrics,
                    "consistencyToReference": {
                        "vectors": vector_consistency,
                        "trialScores": score_consistency,
                    },
                    "resources": {
                        "beforeScan": resources_before_scan,
                        "afterScan": resources_after_scan,
                    },
                }
            )
            if vectors is not reference_vectors:
                del vectors
            if trial_scores is not reference_scores:
                del trial_scores
    finally:
        release_evidence = _release_model_holder(
            model_holder,
            torch_module=torch,
            target_device=target_device,
        )
    resources_after_release = _validated_resource_snapshot(resource_probe)

    retained_allocated = max(
        0,
        resources_after_release["cudaAllocatedBytes"]
        - resources_before_load["cudaAllocatedBytes"],
    )
    retained_reserved = max(
        0,
        resources_after_release["cudaReservedBytes"]
        - resources_before_load["cudaReservedBytes"],
    )
    release_passed = (
        bool(release_evidence["modelObjectCollected"])
        and retained_allocated <= CUDA_RELEASE_TOLERANCE_BYTES
        and retained_reserved <= CUDA_RELEASE_TOLERANCE_BYTES
    )
    consistency_passed = all_vectors_consistent and all_scores_consistent
    source = manifest["source"]
    report = {
        "schemaVersion": "1.1.0",
        "benchmark": "redimnet2-batch-consistency-release",
        "model": model_evidence,
        "wespeaker": source_evidence,
        "trialManifest": {
            "path": str(trial_manifest_path.resolve()),
            "fileSha256": sha256_file(trial_manifest_path.resolve()),
            "canonicalSha256": manifest["canonicalSha256"],
            "sourceAudioSha256": source["audioSha256"],
            "sourceAnnotationSha256": source["annotationSha256"],
            "evaluationSplit": manifest["clips"][0]["evaluationSplit"],
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cudaRuntime": torch.version.cuda,
            "device": str(target_device),
            "deviceName": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else "cpu"
            ),
            "batchSizes": list(normalized_batch_sizes),
            "referenceBatchSize": normalized_batch_sizes[0],
            "absoluteTolerance": absolute_tolerance,
            "relativeTolerance": relative_tolerance,
            "cudaReleaseToleranceBytes": CUDA_RELEASE_TOLERANCE_BYTES,
            "determinism": determinism_evidence,
            "excludedClassificationKeys": sorted(
                EXCLUDED_CLASSIFICATION_KEYS
            ),
        },
        "execution": {
            "speakerCount": manifest["counts"]["speakers"],
            "clipCount": len(clip_rows),
            "trialCount": len(manifest["trials"]),
            "sampleRateHz": sample_rate,
            "modelLoadCount": 1,
            "modelLoadSeconds": model_load_seconds,
            "scans": scans,
        },
        "resources": {
            "beforeModelLoad": resources_before_load,
            "afterModelLoad": resources_after_load,
            "afterRelease": resources_after_release,
            "release": {
                **release_evidence,
                "retainedCudaAllocatedBytes": retained_allocated,
                "retainedCudaReservedBytes": retained_reserved,
                "processRssDeltaFromBeforeLoadBytes": (
                    resources_after_release["processRssBytes"]
                    - resources_before_load["processRssBytes"]
                ),
                "passed": release_passed,
            },
        },
        "validation": {
            "allVectorSetsConsistent": all_vectors_consistent,
            "allTrialScoreSetsConsistent": all_scores_consistent,
            "batchConsistencyPassed": consistency_passed,
            "resourceReleasePassed": release_passed,
            "passed": consistency_passed and release_passed,
        },
        "referenceTrialScores": reference_scores,
    }
    return _attach_canonical_digest(report)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--wespeaker-source", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="comma-separated unique batch sizes (default: 4,8,12,16)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_batch_scan(
        model_path=args.model_path,
        wespeaker_source=args.wespeaker_source,
        trial_manifest_path=args.trial_manifest,
        device=args.device,
        batch_sizes=args.batch_sizes,
    )
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "validation": report["validation"],
                "release": report["resources"]["release"],
                "scans": [
                    {
                        "batchSize": row["batchSize"],
                        "embeddingSeconds": row["embeddingSeconds"],
                        "peakAllocatedVramBytes": row["resources"][
                            "afterScan"
                        ]["cudaPeakAllocatedBytes"],
                        "peakReservedVramBytes": row["resources"][
                            "afterScan"
                        ]["cudaPeakReservedBytes"],
                    }
                    for row in report["execution"]["scans"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
