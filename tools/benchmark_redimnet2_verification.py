"""Benchmark a pinned WeSpeaker ReDimNet2 model on frozen verification trials."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402


EXPECTED_WESPEAKER_COMMIT = "dfa741957e5c11f477623b6e583d67d0af25ee88"
EXPECTED_MODEL_REPO = "Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM"
EXPECTED_MODEL_REVISION = "e34354de2429a45894905bd58c24c22250485b9a"
EXPECTED_WEIGHT_SHA256 = (
    "b9314cd0184d3823c70a2518d354397bf049832f90fb1d7584ff6c0b0d8b152a"
)
REQUIRED_WESPEAKER_FILES = (
    "wespeaker/frontend/tfmel.py",
    "wespeaker/models/pooling_layers.py",
    "wespeaker/models/redimnet2.py",
)
EXCLUDED_CLASSIFICATION_KEYS = frozenset(
    {"projection.bias", "projection.weight"}
)


class ReDimNet2BenchmarkError(ValueError):
    """Raised when frozen benchmark evidence is incomplete or inconsistent."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _canonical_without_newline(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReDimNet2BenchmarkError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ReDimNet2BenchmarkError(f"JSON document is not an object: {path}")
    return value


def load_trial_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    document = _strict_object(resolved)
    expected_digest = document.get("canonicalSha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ReDimNet2BenchmarkError("trial manifest has no canonical digest")
    body = dict(document)
    body.pop("canonicalSha256")
    if canonical_json_sha256(body) != expected_digest:
        raise ReDimNet2BenchmarkError("trial manifest canonical digest mismatches")

    source = document.get("source")
    clips = document.get("clips")
    trials = document.get("trials")
    if not isinstance(source, dict):
        raise ReDimNet2BenchmarkError("trial manifest has no source object")
    if not isinstance(clips, list) or not clips:
        raise ReDimNet2BenchmarkError("trial manifest has no clips")
    if not isinstance(trials, list) or not trials:
        raise ReDimNet2BenchmarkError("trial manifest has no trials")

    audio_text = source.get("audioPath")
    annotation_text = source.get("annotationPath")
    if not isinstance(audio_text, str) or not isinstance(annotation_text, str):
        raise ReDimNet2BenchmarkError("trial source paths are invalid")
    audio_path = Path(audio_text).resolve(strict=True)
    annotation_path = Path(annotation_text).resolve(strict=True)
    if not audio_path.is_file() or not annotation_path.is_file():
        raise ReDimNet2BenchmarkError("trial source artifacts are not files")
    if source.get("audioBytes") != audio_path.stat().st_size:
        raise ReDimNet2BenchmarkError("trial source audio size mismatches")
    if source.get("annotationBytes") != annotation_path.stat().st_size:
        raise ReDimNet2BenchmarkError("trial source annotation size mismatches")
    if source.get("audioSha256") != sha256_file(audio_path):
        raise ReDimNet2BenchmarkError("trial source audio digest mismatches")
    if source.get("annotationSha256") != sha256_file(annotation_path):
        raise ReDimNet2BenchmarkError("trial source annotation digest mismatches")

    clip_by_id: dict[str, Mapping[str, Any]] = {}
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ReDimNet2BenchmarkError(f"clip {index} is not an object")
        clip_id = clip.get("clipId")
        speaker_id = clip.get("speakerId")
        start_ms = clip.get("startMs")
        end_ms = clip.get("endMs")
        if (
            not isinstance(clip_id, str)
            or not clip_id
            or clip_id in clip_by_id
            or not isinstance(speaker_id, str)
            or not speaker_id
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or clip.get("audio") != audio_text
            or clip.get("sourceRecordingId") != source.get("recordingId")
        ):
            raise ReDimNet2BenchmarkError(f"clip {index} is invalid")
        clip_by_id[clip_id] = clip

    trial_ids: set[str] = set()
    class_counts = {True: 0, False: 0}
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ReDimNet2BenchmarkError(f"trial {index} is not an object")
        trial_id = trial.get("trialId")
        left_id = trial.get("enrollmentClipId")
        right_id = trial.get("testClipId")
        same_speaker = trial.get("sameSpeaker")
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or trial_id in trial_ids
            or not isinstance(left_id, str)
            or not isinstance(right_id, str)
            or left_id == right_id
            or left_id not in clip_by_id
            or right_id not in clip_by_id
            or not isinstance(same_speaker, bool)
        ):
            raise ReDimNet2BenchmarkError(f"trial {index} is invalid")
        truth = (
            clip_by_id[left_id]["speakerId"]
            == clip_by_id[right_id]["speakerId"]
        )
        if truth is not same_speaker:
            raise ReDimNet2BenchmarkError(f"trial {index} truth label mismatches")
        trial_ids.add(trial_id)
        class_counts[same_speaker] += 1
    if min(class_counts.values()) < 1:
        raise ReDimNet2BenchmarkError("trial manifest must contain both classes")
    return document


def verify_model_snapshot(model_path: Path) -> dict[str, Any]:
    root = model_path.resolve(strict=True)
    if not root.is_dir():
        raise ReDimNet2BenchmarkError("model path must be a directory")
    manifest_path = root / ".mts-model-manifest.json"
    manifest = _strict_object(manifest_path)
    if (
        manifest.get("provider") != "huggingface"
        or manifest.get("repoId") != EXPECTED_MODEL_REPO
        or manifest.get("revision") != EXPECTED_MODEL_REVISION
    ):
        raise ReDimNet2BenchmarkError("ReDimNet2 model identity mismatches")
    expected_manifest_digest = manifest.get("manifestSha256")
    if not isinstance(expected_manifest_digest, str):
        raise ReDimNet2BenchmarkError("model manifest has no canonical digest")
    body = dict(manifest)
    body.pop("manifestSha256")
    if _canonical_without_newline(body) != expected_manifest_digest:
        raise ReDimNet2BenchmarkError("model manifest canonical digest mismatches")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReDimNet2BenchmarkError("model manifest has no files")
    verified_paths: set[str] = set()
    weight_digest: str | None = None
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise ReDimNet2BenchmarkError(f"model file {index} is invalid")
        relative = row.get("path")
        expected_size = row.get("size")
        expected_sha256 = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in verified_paths
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ReDimNet2BenchmarkError(f"model file {index} metadata is invalid")
        target = (root / relative).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ReDimNet2BenchmarkError("model manifest path escapes root") from exc
        if not target.is_file() or target.stat().st_size != expected_size:
            raise ReDimNet2BenchmarkError(f"model file size mismatches: {relative}")
        actual_sha256 = sha256_file(target)
        if actual_sha256 != expected_sha256:
            raise ReDimNet2BenchmarkError(f"model file digest mismatches: {relative}")
        if relative == "avg_model.pt":
            weight_digest = actual_sha256
        verified_paths.add(relative)
    if weight_digest != EXPECTED_WEIGHT_SHA256:
        raise ReDimNet2BenchmarkError("ReDimNet2 weight digest mismatches")
    return {
        "path": str(root),
        "repoId": manifest["repoId"],
        "revision": manifest["revision"],
        "weightSha256": weight_digest,
        "manifestPath": str(manifest_path),
        "manifestFileSha256": sha256_file(manifest_path),
        "manifestCanonicalSha256": expected_manifest_digest,
        "fileCount": len(files),
        "totalBytes": manifest.get("totalBytes"),
    }


def verify_wespeaker_source(
    source_path: Path,
    *,
    expected_commit: str = EXPECTED_WESPEAKER_COMMIT,
) -> dict[str, Any]:
    root = source_path.resolve(strict=True)
    if not root.is_dir():
        raise ReDimNet2BenchmarkError("WeSpeaker source path must be a directory")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReDimNet2BenchmarkError("WeSpeaker source Git probe failed") from exc
    if commit != expected_commit:
        raise ReDimNet2BenchmarkError(
            f"WeSpeaker commit mismatches: expected {expected_commit}, got {commit}"
        )
    source_files: list[dict[str, Any]] = []
    for relative in REQUIRED_WESPEAKER_FILES:
        local_path = (root / relative).resolve(strict=True)
        try:
            local_path.relative_to(root)
            committed = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{relative}"],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ReDimNet2BenchmarkError(
                f"WeSpeaker source blob probe failed: {relative}"
            ) from exc
        local = local_path.read_bytes()
        if local != committed:
            raise ReDimNet2BenchmarkError(
                f"WeSpeaker required source differs from commit: {relative}"
            )
        source_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(local).hexdigest(),
                "bytes": len(local),
            }
        )
    return {
        "path": str(root),
        "commit": commit,
        "requiredFilesMatchCommit": True,
        "requiredFiles": source_files,
    }


def _install_wespeaker_namespaces(source_path: Path) -> None:
    """Load the pinned core without importing unrelated optional frontends."""

    package_root = source_path / "wespeaker"
    if not package_root.is_dir():
        raise ReDimNet2BenchmarkError("WeSpeaker Python package is missing")
    for name, path in (
        ("wespeaker", package_root),
        ("wespeaker.frontend", package_root / "frontend"),
        ("wespeaker.models", package_root / "models"),
    ):
        existing = sys.modules.get(name)
        if existing is not None:
            existing_paths = tuple(getattr(existing, "__path__", ()))
            if str(path) not in existing_paths:
                raise ReDimNet2BenchmarkError(
                    f"conflicting Python module is already loaded: {name}"
                )
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_model(model_path: Path, source_path: Path, device: str) -> Any:
    try:
        import torch
        import yaml
    except ImportError as exc:
        raise ReDimNet2BenchmarkError("PyTorch and PyYAML are required") from exc
    _install_wespeaker_namespaces(source_path)
    tfmel = importlib.import_module("wespeaker.frontend.tfmel")
    redimnet2 = importlib.import_module("wespeaker.models.redimnet2")
    config = yaml.safe_load(
        (model_path / "config.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(config, dict) or config.get("model") != "ReDimNet2Wrap":
        raise ReDimNet2BenchmarkError("ReDimNet2 config is unsupported")
    model_args = config.get("model_args")
    dataset_args = config.get("dataset_args")
    if not isinstance(model_args, dict) or not isinstance(dataset_args, dict):
        raise ReDimNet2BenchmarkError("ReDimNet2 config sections are missing")
    tfmel_args = dataset_args.get("tfmel_args")
    sample_rate = dataset_args.get("resample_rate")
    if not isinstance(tfmel_args, dict) or sample_rate != 16_000:
        raise ReDimNet2BenchmarkError("ReDimNet2 frontend config is unsupported")

    frontend_options = dict(tfmel_args)
    frontend_options.setdefault("sample_rate", sample_rate)
    model = redimnet2.ReDimNet2Wrap(**model_args)
    model.add_module("frontend", tfmel.TFMelFrontend(**frontend_options))
    model.prepare_for_frontend("tfmel")
    state = torch.load(
        model_path / "avg_model.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(state, Mapping):
        raise ReDimNet2BenchmarkError("ReDimNet2 checkpoint is not a state dict")
    checkpoint = dict(state)
    present_excluded = {
        key for key in checkpoint if key.startswith("projection.")
    }
    if present_excluded != EXCLUDED_CLASSIFICATION_KEYS:
        raise ReDimNet2BenchmarkError(
            "ReDimNet2 checkpoint classification keys are unexpected"
        )
    for key in EXCLUDED_CLASSIFICATION_KEYS:
        checkpoint.pop(key)
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as exc:
        raise ReDimNet2BenchmarkError("ReDimNet2 checkpoint is incompatible") from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ReDimNet2BenchmarkError("CUDA was requested but is unavailable")
    return model.to(target_device).eval()


def verification_metrics(
    scores: Sequence[tuple[float, bool]],
) -> dict[str, float]:
    positives = [score for score, label in scores if label]
    negatives = [score for score, label in scores if not label]
    if not positives or not negatives:
        raise ReDimNet2BenchmarkError("both verification classes are required")
    if not all(math.isfinite(score) for score, _ in scores):
        raise ReDimNet2BenchmarkError("verification score is non-finite")

    unique = sorted({score for score, _ in scores}, reverse=True)
    thresholds = [math.inf]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    )
    thresholds.append(-math.inf)
    operating_points: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        false_reject = sum(score < threshold for score in positives) / len(positives)
        false_accept = sum(score >= threshold for score in negatives) / len(negatives)
        operating_points.append((threshold, false_reject, false_accept))
    threshold, false_reject, false_accept = min(
        operating_points,
        key=lambda point: (
            abs(point[1] - point[2]),
            (point[1] + point[2]) / 2.0,
        ),
    )
    min_dcf = min(
        0.01 * miss + 0.99 * false_alarm
        for _, miss, false_alarm in operating_points
    )
    auc_numerator = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    auc = auc_numerator / (len(positives) * len(negatives))
    true_accept = sum(score >= threshold for score in positives) / len(positives)
    true_reject = sum(score < threshold for score in negatives) / len(negatives)
    return {
        "eer": (false_reject + false_accept) / 2.0,
        "eerThreshold": threshold,
        "falseRejectRateAtEer": false_reject,
        "falseAcceptRateAtEer": false_accept,
        "balancedAccuracyAtEer": (true_accept + true_reject) / 2.0,
        "auc": auc,
        "minDcfPTarget0.01": min_dcf,
        "positiveMeanCosine": sum(positives) / len(positives),
        "negativeMeanCosine": sum(negatives) / len(negatives),
        "meanCosineSeparation": (
            sum(positives) / len(positives)
            - sum(negatives) / len(negatives)
        ),
    }


def _embedding_set_sha256(
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
    if not vectors:
        raise ReDimNet2BenchmarkError("embedding set is empty")
    return digest.hexdigest()


def _process_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def run_benchmark(
    *,
    model_path: Path,
    wespeaker_source: Path,
    trial_manifest_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ReDimNet2BenchmarkError("batch size must be positive")
    model_evidence = verify_model_snapshot(model_path)
    source_evidence = verify_wespeaker_source(wespeaker_source)
    manifest = load_trial_manifest(trial_manifest_path)

    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise ReDimNet2BenchmarkError(
            "NumPy, SoundFile, and PyTorch are required"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ReDimNet2BenchmarkError("CUDA benchmark requested but unavailable")
    target_device = torch.device(device)
    memory_before = _process_memory_bytes()
    load_started = time.perf_counter()
    model = _load_model(model_path.resolve(), wespeaker_source.resolve(), device)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        model_vram_bytes = int(torch.cuda.memory_allocated(target_device))
        torch.cuda.reset_peak_memory_stats(target_device)
    else:
        model_vram_bytes = 0
    load_seconds = time.perf_counter() - load_started

    source = manifest["source"]
    audio, sample_rate = sf.read(
        Path(str(source["audioPath"])), dtype="float32", always_2d=True
    )
    if sample_rate != source["audio"]["sampleRateHz"] or audio.shape[1] != 1:
        raise ReDimNet2BenchmarkError("benchmark currently requires mono source audio")
    clip_rows = sorted(manifest["clips"], key=lambda row: str(row["clipId"]))
    clip_waves: list[Any] = []
    expected_samples: int | None = None
    for clip in clip_rows:
        start_sample = round(int(clip["startMs"]) * sample_rate / 1000.0)
        end_sample = round(int(clip["endMs"]) * sample_rate / 1000.0)
        wave = np.asarray(audio[start_sample:end_sample, 0], dtype=np.float32)
        if wave.size < 1:
            raise ReDimNet2BenchmarkError("trial clip is empty")
        if expected_samples is None:
            expected_samples = int(wave.size)
        elif wave.size != expected_samples:
            raise ReDimNet2BenchmarkError("trial clips do not have equal duration")
        clip_waves.append(wave)

    vectors: dict[str, Any] = {}
    embed_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(clip_rows), batch_size):
            rows = clip_rows[offset : offset + batch_size]
            batch = torch.from_numpy(
                np.stack(clip_waves[offset : offset + batch_size])
            ).to(target_device)
            lengths = torch.full(
                (batch.shape[0],),
                batch.shape[1],
                dtype=torch.long,
                device=target_device,
            )
            features, _ = model.frontend(batch, lengths)
            embeddings = torch.nn.functional.normalize(
                model(features).float(), dim=-1
            ).cpu()
            for row, vector in zip(rows, embeddings):
                vectors[str(row["clipId"])] = vector
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(target_device))
        peak_reserved_vram_bytes = int(
            torch.cuda.max_memory_reserved(target_device)
        )
    else:
        peak_vram_bytes = 0
        peak_reserved_vram_bytes = 0
    embedding_seconds = time.perf_counter() - embed_started
    memory_after = _process_memory_bytes()

    trial_scores: list[dict[str, Any]] = []
    metric_input: list[tuple[float, bool]] = []
    for trial in manifest["trials"]:
        left = vectors[str(trial["enrollmentClipId"])]
        right = vectors[str(trial["testClipId"])]
        score = float(torch.dot(left, right).item())
        same_speaker = bool(trial["sameSpeaker"])
        metric_input.append((score, same_speaker))
        trial_scores.append(
            {
                "trialId": trial["trialId"],
                "enrollmentClipId": trial["enrollmentClipId"],
                "testClipId": trial["testClipId"],
                "sameSpeaker": same_speaker,
                "cosine": score,
            }
        )
    metrics = verification_metrics(metric_input)
    dimension_set = {int(vector.shape[-1]) for vector in vectors.values()}
    if len(dimension_set) != 1:
        raise ReDimNet2BenchmarkError("embedding dimensions are inconsistent")

    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "benchmark": "redimnet2-frozen-speaker-verification",
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
            "batchSize": batch_size,
            "excludedClassificationKeys": sorted(EXCLUDED_CLASSIFICATION_KEYS),
        },
        "execution": {
            "speakerCount": manifest["counts"]["speakers"],
            "clipCount": len(clip_rows),
            "trialCount": len(trial_scores),
            "embeddingDimensions": next(iter(dimension_set)),
            "embeddingSetHashAlgorithm": "clip-id-sorted-float32-le-v1",
            "embeddingSetSha256": _embedding_set_sha256(
                vectors, numpy_module=np
            ),
            "modelLoadSeconds": load_seconds,
            "embeddingSeconds": embedding_seconds,
            "embeddingSecondsPerClip": embedding_seconds / len(clip_rows),
            "modelVramBytes": model_vram_bytes,
            "peakAllocatedVramBytes": peak_vram_bytes,
            "peakReservedVramBytes": peak_reserved_vram_bytes,
            "processRssBeforeBytes": memory_before,
            "processRssAfterBytes": memory_after,
        },
        "metrics": metrics,
        "trialScores": trial_scores,
    }
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


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
    parser.add_argument("--batch-size", type=_positive_integer, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        model_path=args.model_path,
        wespeaker_source=args.wespeaker_source,
        trial_manifest_path=args.trial_manifest,
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
