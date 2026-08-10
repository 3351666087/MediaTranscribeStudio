"""Benchmark the pinned official Palabra ReDimNet2-B6-LM checkpoint."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from tools.benchmark_redimnet2_verification import (  # noqa: E402
    _process_memory_bytes,
    _write_report,
    load_trial_manifest,
    verification_metrics,
)


EXPECTED_REPOSITORY = "PalabraAI/redimnet2"
EXPECTED_TAG = "v1.0.0"
EXPECTED_TAG_OBJECT = "b318ea063fcfe63df7d8d4531c831d9116852808"
EXPECTED_COMMIT = "5294667e806ac3b0f27abc301a114ef132b64b42"
EXPECTED_CHECKPOINT_BYTES = 50_881_144
EXPECTED_CHECKPOINT_SHA256 = (
    "bc45f032099f3c9f0fb6ad756bca23daeeba456d00c515056e388824ef36f41b"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "5167e54c405775ac15246f744c86e017cbbc84a187f536e10416919a069c49cb"
)
EXPECTED_TENSOR_COUNT = 807
EXPECTED_PARAMETER_COUNT = 12_656_496
DEFAULT_CANDIDATE_LOCK = (
    PROJECT_ROOT
    / "benchmarks"
    / "speaker_models"
    / "palabra-redimnet2-b6-lm.candidate.lock.json"
)
REQUIRED_SOURCE_FILES = (
    "hubconf.py",
    "redimnet2/__init__.py",
    "redimnet2/redimnet2.py",
    "redimnet2/layers/attention.py",
    "redimnet2/layers/blocks.py",
    "redimnet2/layers/convnext.py",
    "redimnet2/layers/features.py",
    "redimnet2/layers/features_tf.py",
    "redimnet2/layers/layernorm.py",
    "redimnet2/layers/poolings.py",
    "redimnet2/layers/redim_structural.py",
    "redimnet2/layers/resblocks.py",
)


class PalabraReDimNet2BenchmarkError(RuntimeError):
    """Raised when official-model evidence or inference is inconsistent."""


def _initialize_cuda_measurement(torch: Any, target: Any) -> None:
    """Prime the target CUDA context before resetting peak-memory counters."""

    torch.cuda.get_device_properties(target)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(target)


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PalabraReDimNet2BenchmarkError("source Git probe failed") from exc


def verify_source_snapshot(
    source_path: Path,
    *,
    expected_commit: str = EXPECTED_COMMIT,
    expected_tag: str = EXPECTED_TAG,
    expected_tag_object: str = EXPECTED_TAG_OBJECT,
    required_files: Sequence[str] = REQUIRED_SOURCE_FILES,
) -> dict[str, Any]:
    root = source_path.resolve(strict=True)
    if not root.is_dir():
        raise PalabraReDimNet2BenchmarkError("source path must be a directory")
    commit = _git_output(root, "rev-parse", "HEAD")
    tag_object = _git_output(root, "rev-parse", expected_tag)
    tag_commit = _git_output(root, "rev-parse", f"{expected_tag}^{{commit}}")
    if commit != expected_commit or tag_commit != commit:
        raise PalabraReDimNet2BenchmarkError("source commit or tag mismatches")
    if tag_object != expected_tag_object:
        raise PalabraReDimNet2BenchmarkError("source tag object mismatches")
    files: list[dict[str, Any]] = []
    for relative in required_files:
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
            committed = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{relative}"],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise PalabraReDimNet2BenchmarkError(
                f"source file probe failed: {relative}"
            ) from exc
        local = path.read_bytes()
        if local != committed:
            raise PalabraReDimNet2BenchmarkError(
                f"source file differs from commit: {relative}"
            )
        files.append(
            {
                "path": relative,
                "bytes": len(local),
                "sha256": sha256_file(path),
            }
        )
    return {
        "path": str(root),
        "repository": EXPECTED_REPOSITORY,
        "commit": commit,
        "tag": expected_tag,
        "tagObject": tag_object,
        "requiredFiles": files,
        "requiredFilesMatchCommit": True,
    }


def verify_checkpoint(
    checkpoint_path: Path,
    candidate_lock_path: Path = DEFAULT_CANDIDATE_LOCK,
) -> dict[str, Any]:
    checkpoint = checkpoint_path.resolve(strict=True)
    lock = candidate_lock_path.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise PalabraReDimNet2BenchmarkError(
            "checkpoint must be a regular local file"
        )
    if (
        checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES
        or sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256
    ):
        raise PalabraReDimNet2BenchmarkError("checkpoint identity mismatches")
    try:
        document = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PalabraReDimNet2BenchmarkError("candidate lock is invalid") from exc
    model = document.get("model") if isinstance(document, dict) else None
    license_evidence = document.get("license") if isinstance(document, dict) else None
    if (
        not isinstance(model, dict)
        or model.get("assetBytes") != EXPECTED_CHECKPOINT_BYTES
        or model.get("assetSha256") != EXPECTED_CHECKPOINT_SHA256
        or model.get("commit") != EXPECTED_COMMIT
        or not isinstance(license_evidence, dict)
        or license_evidence.get("productionPromotionAllowed") is not False
    ):
        raise PalabraReDimNet2BenchmarkError("candidate lock mismatches")
    return {
        "path": str(checkpoint),
        "repoId": EXPECTED_REPOSITORY,
        "revision": EXPECTED_COMMIT,
        "releaseTag": EXPECTED_TAG,
        "weightBytes": checkpoint.stat().st_size,
        "weightSha256": EXPECTED_CHECKPOINT_SHA256,
        "manifestPath": str(lock),
        "manifestFileSha256": sha256_file(lock),
        "licenseDeclared": license_evidence.get("declaredSpdx"),
        "licenseTextFilePresentAtPinnedTag": False,
        "productionPromotionAllowed": False,
    }


def _load_model(checkpoint_path: Path, source_path: Path, device: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise PalabraReDimNet2BenchmarkError("PyTorch is required") from exc
    root = source_path.resolve()
    loaded = sys.modules.get("redimnet2")
    if loaded is not None:
        loaded_paths = tuple(str(item) for item in getattr(loaded, "__path__", ()))
        if str(root / "redimnet2") not in loaded_paths:
            raise PalabraReDimNet2BenchmarkError(
                "conflicting redimnet2 module is already loaded"
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("redimnet2.redimnet2")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_config",
        "state_dict",
    }:
        raise PalabraReDimNet2BenchmarkError("checkpoint structure mismatches")
    config = checkpoint["model_config"]
    state = checkpoint["state_dict"]
    if (
        not isinstance(config, dict)
        or canonical_json_sha256(config) != EXPECTED_MODEL_CONFIG_SHA256
        or not isinstance(state, Mapping)
        or len(state) != EXPECTED_TENSOR_COUNT
    ):
        raise PalabraReDimNet2BenchmarkError("checkpoint metadata mismatches")
    parameter_count = sum(
        int(value.numel()) for value in state.values() if hasattr(value, "numel")
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise PalabraReDimNet2BenchmarkError("checkpoint parameter count mismatches")
    model = module.ReDimNet2Wrap(**config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PalabraReDimNet2BenchmarkError(
            "checkpoint tensors are incompatible"
        ) from exc
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise PalabraReDimNet2BenchmarkError("CUDA was requested but unavailable")
    return model.to(target).eval()


def run_benchmark(
    *,
    checkpoint_path: Path,
    source_path: Path,
    trial_manifest_path: Path,
    candidate_lock_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_evidence = verify_checkpoint(checkpoint_path, candidate_lock_path)
    source_evidence = verify_source_snapshot(source_path)
    manifest = load_trial_manifest(trial_manifest_path)
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise PalabraReDimNet2BenchmarkError(
            "NumPy, SoundFile, and PyTorch are required"
        ) from exc
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise PalabraReDimNet2BenchmarkError("CUDA benchmark requested but unavailable")

    memory_before = _process_memory_bytes()
    if target.type == "cuda":
        _initialize_cuda_measurement(torch, target)
        allocated_before = int(torch.cuda.memory_allocated(target))
        reserved_before = int(torch.cuda.memory_reserved(target))
    else:
        allocated_before = 0
        reserved_before = 0
    load_started = time.perf_counter()
    model = _load_model(checkpoint_path.resolve(), source_path.resolve(), device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        model_vram_bytes = int(torch.cuda.memory_allocated(target)) - allocated_before
        torch.cuda.reset_peak_memory_stats(target)
    else:
        model_vram_bytes = 0
    load_seconds = time.perf_counter() - load_started

    source = manifest["source"]
    audio, sample_rate = sf.read(
        Path(str(source["audioPath"])), dtype="float32", always_2d=True
    )
    if sample_rate != source["audio"]["sampleRateHz"] or audio.shape[1] != 1:
        raise PalabraReDimNet2BenchmarkError("benchmark requires mono source audio")
    clip_rows = sorted(manifest["clips"], key=lambda row: str(row["clipId"]))
    clip_waves: list[Any] = []
    expected_samples: int | None = None
    for clip in clip_rows:
        start = round(int(clip["startMs"]) * sample_rate / 1000.0)
        end = round(int(clip["endMs"]) * sample_rate / 1000.0)
        wave = np.asarray(audio[start:end, 0], dtype=np.float32)
        if wave.size < 1:
            raise PalabraReDimNet2BenchmarkError("trial clip is empty")
        if expected_samples is None:
            expected_samples = int(wave.size)
        elif wave.size != expected_samples:
            raise PalabraReDimNet2BenchmarkError(
                "trial clips must have equal duration"
            )
        clip_waves.append(wave)

    vectors: dict[str, Any] = {}
    embed_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(clip_rows), batch_size):
            rows = clip_rows[offset : offset + batch_size]
            batch = torch.from_numpy(
                np.stack(clip_waves[offset : offset + batch_size])
            ).to(target)
            embeddings = torch.nn.functional.normalize(
                model(batch).float(), dim=-1
            ).cpu()
            for row, vector in zip(rows, embeddings):
                vectors[str(row["clipId"])] = vector
            del batch, embeddings
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        peak_allocated = int(torch.cuda.max_memory_allocated(target))
        peak_reserved = int(torch.cuda.max_memory_reserved(target))
    else:
        peak_allocated = 0
        peak_reserved = 0
    embedding_seconds = time.perf_counter() - embed_started

    trial_scores: list[dict[str, Any]] = []
    metric_input: list[tuple[float, bool]] = []
    for trial in manifest["trials"]:
        score = float(
            torch.dot(
                vectors[str(trial["enrollmentClipId"])],
                vectors[str(trial["testClipId"])],
            ).item()
        )
        if not math.isfinite(score):
            raise PalabraReDimNet2BenchmarkError("trial score is non-finite")
        same = bool(trial["sameSpeaker"])
        metric_input.append((score, same))
        trial_scores.append(
            {
                "trialId": trial["trialId"],
                "enrollmentClipId": trial["enrollmentClipId"],
                "testClipId": trial["testClipId"],
                "sameSpeaker": same,
                "cosine": score,
            }
        )
    dimensions = {int(vector.shape[-1]) for vector in vectors.values()}
    if dimensions != {192}:
        raise PalabraReDimNet2BenchmarkError("embedding dimensions mismatch")
    metrics = verification_metrics(metric_input)
    memory_after_inference = _process_memory_bytes()

    del model, audio, clip_waves, vectors
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(target)
        allocated_after_release = int(torch.cuda.memory_allocated(target))
        reserved_after_release = int(torch.cuda.memory_reserved(target))
    else:
        allocated_after_release = 0
        reserved_after_release = 0
    memory_after_release = _process_memory_bytes()

    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "benchmark": "redimnet2-frozen-speaker-verification",
        "promotionEligible": False,
        "promotionBlockers": [
            "pinned tag has no standalone license text",
            "development evidence cannot establish held-out non-regression",
        ],
        "model": model_evidence,
        "sourceImplementation": source_evidence,
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
            "device": str(target),
            "deviceName": (
                torch.cuda.get_device_name(target)
                if target.type == "cuda"
                else "cpu"
            ),
            "batchSize": batch_size,
        },
        "execution": {
            "speakerCount": manifest["counts"]["speakers"],
            "clipCount": len(clip_rows),
            "trialCount": len(trial_scores),
            "embeddingDimensions": 192,
            "modelLoadSeconds": load_seconds,
            "embeddingSeconds": embedding_seconds,
            "embeddingSecondsPerClip": embedding_seconds / len(clip_rows),
            "modelVramBytes": model_vram_bytes,
            "peakAllocatedVramBytes": peak_allocated,
            "peakReservedVramBytes": peak_reserved,
            "processRssBeforeBytes": memory_before,
            "processRssAfterInferenceBytes": memory_after_inference,
            "processRssAfterReleaseBytes": memory_after_release,
            "cudaAllocatedBeforeBytes": allocated_before,
            "cudaReservedBeforeBytes": reserved_before,
            "cudaAllocatedAfterReleaseBytes": allocated_after_release,
            "cudaReservedAfterReleaseBytes": reserved_after_release,
            "resourcesReleased": (
                allocated_after_release <= allocated_before
                and reserved_after_release <= reserved_before
            ),
        },
        "metrics": metrics,
        "trialScores": trial_scores,
    }
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_integer, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        checkpoint_path=args.checkpoint,
        source_path=args.source,
        trial_manifest_path=args.trial_manifest,
        candidate_lock_path=args.candidate_lock,
        device=args.device,
        batch_size=args.batch_size,
    )
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "metrics": report["metrics"],
                "execution": report["execution"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
