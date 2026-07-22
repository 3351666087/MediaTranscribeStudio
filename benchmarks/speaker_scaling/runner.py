"""Runner for the existing backend speaker-clustering implementation."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import os
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import speaker_pipeline
from backend.models import SpeakerCountPolicy, StartJobRequest
from backend.speaker_pipeline import SpeakerPipelineConfig

from .synthetic import SCENARIO_NAMES, SyntheticCase, generate_scenario


DEFAULT_SPEAKER_COUNTS = (1, 2, 3, 5, 8, 13, 32, 64, 129)
DEFAULT_SAMPLES_PER_SPEAKER = 3
DEFAULT_MODES = ("manual", "auto", "hybrid")
ALGORITHM_ENTRY_POINT = "backend.speaker_pipeline._cluster"


def benchmark_config() -> SpeakerPipelineConfig:
    """Return the explicit, reportable configuration used by this benchmark."""

    return SpeakerPipelineConfig(
        cluster_similarity_threshold=0.72,
        max_auto_speakers=None,
        max_clustering_windows=10_000_000,
        max_clustering_work_items=2_000_000_000,
        kmeans_iterations=2,
        max_count_uncertainty_candidates=9,
        auto_count_confidence_threshold=0.70,
    )


def _request(mode: str, expected_count: int) -> StartJobRequest:
    payload: dict[str, object] = {"speakerCountMode": mode}
    if mode == "manual":
        payload["speakerCount"] = expected_count
    elif mode == "hybrid":
        payload["speakerCountBounds"] = {
            "min": max(1, expected_count - 2),
            "max": expected_count + 2,
        }
        payload["speakerCountPrior"] = expected_count
    elif mode != "auto":
        raise ValueError("mode must be manual, auto, or hybrid")
    return StartJobRequest(
        job_id=f"speaker-scaling-{mode}-{expected_count}",
        source_path=Path("synthetic-speaker-scaling.no-media"),
        output_directory=Path("synthetic-speaker-scaling.no-output"),
        speaker_policy=SpeakerCountPolicy.from_payload(payload),
    )


def partition_is_correct(
    case: SyntheticCase,
    assignments: Sequence[int],
    predicted_count: int,
) -> bool:
    """Check exact persistent-speaker partition, ignoring label permutation."""

    if len(assignments) != len(case.windows):
        return False
    if predicted_count != case.persistent_speaker_count:
        return False

    cluster_by_truth: dict[int, set[int]] = {}
    for window, assignment in zip(case.windows, assignments):
        truth = case.truth_by_window_id[window.window_id]
        if truth is None:
            continue
        cluster_by_truth.setdefault(truth, set()).add(int(assignment))

    if set(cluster_by_truth) != set(range(case.persistent_speaker_count)):
        return False
    if any(len(cluster_ids) != 1 for cluster_ids in cluster_by_truth.values()):
        return False
    persistent_clusters = {
        next(iter(cluster_ids)) for cluster_ids in cluster_by_truth.values()
    }
    return len(persistent_clusters) == case.persistent_speaker_count


def _candidate_dict(candidate: Any) -> dict[str, object]:
    if hasattr(candidate, "as_dict"):
        return dict(candidate.as_dict())
    return {
        "count": int(candidate.count),
        "workItems": int(candidate.work_items),
    }


def _summary(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    wall_times = [float(run["wallTimeSeconds"]) for run in runs]
    throughputs = [float(run["windowsPerSecond"]) for run in runs]
    confidences = [float(run["confidence"]) for run in runs]
    predicted = [int(run["predictedSpeakerCount"]) for run in runs]
    total_work = [
        int(dict(run["workItems"])["total"])  # type: ignore[arg-type]
        for run in runs
    ]
    return {
        "minimumWallTimeSeconds": min(wall_times),
        "medianWallTimeSeconds": statistics.median(wall_times),
        "maximumWallTimeSeconds": max(wall_times),
        "medianWindowsPerSecond": statistics.median(throughputs),
        "medianConfidence": statistics.median(confidences),
        "medianTotalWorkItems": statistics.median(total_work),
        "allPartitionsCorrect": all(
            bool(run["partitionCorrect"]) for run in runs
        ),
        "predictionsStable": len(set(predicted)) == 1,
        "predictedSpeakerCounts": predicted,
    }


def run_case(
    case: SyntheticCase,
    *,
    mode: str,
    repeat: int,
    config: SpeakerPipelineConfig | None = None,
    cluster_fn: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Benchmark one synthetic case and count-policy mode."""

    if repeat < 1:
        raise ValueError("repeat must be a positive integer")
    active_config = config or benchmark_config()
    active_cluster = cluster_fn or speaker_pipeline._cluster
    request = _request(mode, case.persistent_speaker_count)
    runs: list[dict[str, object]] = []

    for repeat_index in range(repeat):
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        started_ns = time.perf_counter_ns()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            result = active_cluster(
                case.embeddings,
                case.windows,
                request,
                active_config,
            )
        elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
        wall_time = elapsed_ns / 1_000_000_000.0
        candidates = tuple(result.count_candidates)
        run: dict[str, object] = {
            "repeatIndex": repeat_index,
            "wallTimeSeconds": wall_time,
            "windowsPerSecond": len(case.windows) / wall_time,
            "predictedSpeakerCount": int(result.count),
            "candidateRange": {
                "min": int(result.candidate_min),
                "max": int(result.candidate_max),
            },
            "confidence": float(result.confidence),
            "leaderEstimate": (
                int(result.leader_estimate)
                if result.leader_estimate is not None
                else None
            ),
            "countSearchTruncated": bool(result.count_search_truncated),
            "workItems": {
                "leaderCount": int(result.leader_count_work_items),
                "candidateSearch": sum(
                    int(candidate.work_items) for candidate in candidates
                ),
                "total": int(result.total_work_items),
            },
            "countCandidates": [
                _candidate_dict(candidate) for candidate in candidates
            ],
            "partitionCorrect": partition_is_correct(
                case,
                result.assignments,
                int(result.count),
            ),
        }
        captured = {
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
        }
        if captured["stdout"] or captured["stderr"]:
            run["capturedDiagnostics"] = captured
        runs.append(run)

    return {
        "scenario": case.scenario,
        "mode": mode,
        "expectedPersistentSpeakerCount": case.persistent_speaker_count,
        "windowCount": len(case.windows),
        "embeddingDimension": case.embedding_dimension,
        "inputOrder": (
            "chronological" if case.input_is_time_ordered else "shuffled"
        ),
        "singletonOutlierCount": len(case.outlier_window_ids),
        "outlierExcludedFromPersistentSpeakerTruth": bool(
            case.outlier_window_ids
        ),
        "nearVoiceCosine": case.near_voice_cosine,
        "metric": "exactSyntheticPartition",
        "isRealDer": False,
        "runs": runs,
        "summary": _summary(runs),
    }


def _resolve_method() -> tuple[str, str]:
    try:
        source = inspect.getsource(speaker_pipeline.SpeakerPipeline.transcribe)
    except (OSError, TypeError):
        return ALGORITHM_ENTRY_POINT, "fallback-entry-point"
    match = re.search(r'\bmethod\s*=\s*["\']([^"\']+)["\']', source)
    if match is None:
        return ALGORITHM_ENTRY_POINT, "fallback-entry-point"
    return match.group(1), "SpeakerPipeline.transcribe declaration"


def _algorithm_report(config: SpeakerPipelineConfig) -> dict[str, object]:
    module_path = Path(str(speaker_pipeline.__file__)).resolve()
    method, method_source = _resolve_method()
    return {
        "entryPoint": ALGORITHM_ENTRY_POINT,
        "method": method,
        "methodResolution": method_source,
        "modulePath": str(module_path),
        "moduleSha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "config": {
            "clusterSimilarityThreshold": config.cluster_similarity_threshold,
            "maxAutoSpeakers": config.max_auto_speakers,
            "maxClusteringWindows": config.max_clustering_windows,
            "maxClusteringWorkItems": config.max_clustering_work_items,
            "kmeansIterations": config.kmeans_iterations,
            "maxCountUncertaintyCandidates": (
                config.max_count_uncertainty_candidates
            ),
            "autoCountConfidenceThreshold": (
                config.auto_count_confidence_threshold
            ),
        },
    }

def _environment_report() -> dict[str, object]:
    return {
        "pythonVersion": platform.python_version(),
        "pythonImplementation": platform.python_implementation(),
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
    }


def run_benchmark(
    *,
    speaker_counts: Sequence[int] = DEFAULT_SPEAKER_COUNTS,
    samples_per_speaker: int = DEFAULT_SAMPLES_PER_SPEAKER,
    modes: Sequence[str] = DEFAULT_MODES,
    repeat: int = 1,
    scenario_names: Sequence[str] = SCENARIO_NAMES,
) -> dict[str, object]:
    """Run the requested matrix and return a JSON-serializable report."""

    config = benchmark_config()
    cases: list[dict[str, object]] = []
    for speaker_count in speaker_counts:
        for scenario in scenario_names:
            synthetic = generate_scenario(
                scenario,
                speaker_count=int(speaker_count),
                samples_per_speaker=samples_per_speaker,
            )
            for mode in modes:
                cases.append(
                    run_case(
                        synthetic,
                        mode=mode,
                        repeat=repeat,
                        config=config,
                    )
                )

    run_count = sum(len(list(case["runs"])) for case in cases)
    return {
        "schemaVersion": "speaker-scaling-benchmark/v1",
        "benchmarkKind": "synthetic-speaker-clustering-scaling",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": {
            "loadsModels": False,
            "loadsMedia": False,
            "isRealDer": False,
            "metric": "exactSyntheticPartition",
            "timingScope": "backend.speaker_pipeline._cluster only",
        },
        "environment": _environment_report(),
        "algorithm": _algorithm_report(config),
        "parameters": {
            "speakerCounts": [int(value) for value in speaker_counts],
            "samplesPerSpeaker": samples_per_speaker,
            "modes": list(modes),
            "repeat": repeat,
            "scenarios": list(scenario_names),
        },
        "cases": cases,
        "summary": {
            "caseCount": len(cases),
            "runCount": run_count,
            "allPartitionsCorrect": all(
                bool(dict(case["summary"])["allPartitionsCorrect"])
                for case in cases
            ),
            "correctCaseCount": sum(
                bool(dict(case["summary"])["allPartitionsCorrect"])
                for case in cases
            ),
        },
    }
