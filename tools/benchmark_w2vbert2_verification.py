"""Benchmark the pinned VoxCeleb-only W2V-BERT2 challenger."""

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
from tools.benchmark_redimnet2_verification import (  # noqa: E402
    load_trial_manifest,
    verification_metrics,
)


EXPECTED_CANDIDATE_ID = "w2vbert2-voxceleb-lm"
EXPECTED_MODELSCOPE_REVISION = "43838cdfb13a05fee1d2b451435128f0515f775e"
EXPECTED_WESPEAKER_COMMIT = "dfa741957e5c11f477623b6e583d67d0af25ee88"
EXPECTED_BASE_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
EXCLUDED_CLASSIFICATION_KEYS = frozenset({"projection.weight"})
REQUIRED_WESPEAKER_FILES = (
    "wespeaker/models/pooling_layers.py",
    "wespeaker/models/w2vbert_adapter_mfa.py",
)
DEFAULT_LOCK = (
    PROJECT_ROOT
    / "benchmarks"
    / "speaker_models"
    / "w2vbert2-voxceleb-lm.candidate.lock.json"
)


class W2VBert2BenchmarkError(ValueError):
    """Raised when candidate evidence or runtime behavior is invalid."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _strict_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise W2VBert2BenchmarkError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise W2VBert2BenchmarkError(f"JSON document is not an object: {path}")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise W2VBert2BenchmarkError(f"{field} is not SHA-256")
    return value


def _verify_declared_files(
    root: Path,
    rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise W2VBert2BenchmarkError("candidate lock has no files")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise W2VBert2BenchmarkError(f"candidate file {index} is invalid")
        relative = row.get("path")
        expected_bytes = row.get("bytes")
        expected_sha256 = _sha256_text(
            row.get("sha256"), f"candidate file {index} sha256"
        )
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 1
        ):
            raise W2VBert2BenchmarkError(
                f"candidate file {index} metadata is invalid"
            )
        target = (root / relative).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise W2VBert2BenchmarkError(
                f"candidate file escapes model root: {relative}"
            ) from exc
        actual_bytes = target.stat().st_size
        actual_sha256 = sha256_file(target)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise W2VBert2BenchmarkError(
                f"candidate file no longer matches lock: {relative}"
            )
        seen.add(relative)
        verified.append(
            {
                "path": relative,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    return verified


def verify_candidate_snapshot(
    model_path: Path,
    candidate_lock_path: Path,
) -> dict[str, Any]:
    root = model_path.resolve(strict=True)
    if not root.is_dir():
        raise W2VBert2BenchmarkError("candidate model path must be a directory")
    lock_path = candidate_lock_path.resolve(strict=True)
    lock = _strict_object(lock_path)
    declared_canonical = _sha256_text(
        lock.get("canonicalSha256"), "candidate lock canonicalSha256"
    )
    body = dict(lock)
    body.pop("canonicalSha256")
    if canonical_json_sha256(body) != declared_canonical:
        raise W2VBert2BenchmarkError("candidate lock canonical digest mismatches")
    source = lock.get("source")
    base = lock.get("baseModel")
    wespeaker = lock.get("wespeaker")
    if (
        lock.get("schemaVersion") != "1.0.0"
        or lock.get("candidateId") != EXPECTED_CANDIDATE_ID
        or lock.get("status") != "research-challenger"
        or lock.get("modelPromotionAllowed") is not False
        or not isinstance(source, dict)
        or source.get("revision") != EXPECTED_MODELSCOPE_REVISION
        or not isinstance(base, dict)
        or base.get("revision") != EXPECTED_BASE_REVISION
        or not isinstance(wespeaker, dict)
        or wespeaker.get("revision") != EXPECTED_WESPEAKER_COMMIT
    ):
        raise W2VBert2BenchmarkError("candidate lock identity or policy mismatches")
    verified_files = _verify_declared_files(root, lock.get("files"))
    checkpoint = next(
        (
            row
            for row in verified_files
            if row["path"] == source.get("checkpoint")
        ),
        None,
    )
    if (
        checkpoint is None
        or checkpoint["bytes"] != source.get("checkpointBytes")
        or checkpoint["sha256"] != source.get("checkpointSha256")
    ):
        raise W2VBert2BenchmarkError("candidate checkpoint identity mismatches")
    return {
        "path": str(root),
        "candidateId": lock["candidateId"],
        "status": lock["status"],
        "modelPromotionAllowed": lock["modelPromotionAllowed"],
        "source": source,
        "baseModel": base,
        "licenseReview": lock.get("licenseReview"),
        "lockPath": str(lock_path),
        "lockFileSha256": sha256_file(lock_path),
        "lockCanonicalSha256": declared_canonical,
        "files": verified_files,
    }


def verify_wespeaker_source(source_path: Path) -> dict[str, Any]:
    root = source_path.resolve(strict=True)
    if not root.is_dir():
        raise W2VBert2BenchmarkError("WeSpeaker source path must be a directory")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise W2VBert2BenchmarkError("WeSpeaker source Git probe failed") from exc
    if commit != EXPECTED_WESPEAKER_COMMIT:
        raise W2VBert2BenchmarkError("WeSpeaker source revision mismatches")
    evidence: list[dict[str, Any]] = []
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
            raise W2VBert2BenchmarkError(
                f"cannot verify WeSpeaker source file: {relative}"
            ) from exc
        local = local_path.read_bytes()
        if local != committed:
            raise W2VBert2BenchmarkError(
                f"WeSpeaker source differs from commit: {relative}"
            )
        evidence.append(
            {
                "path": relative,
                "bytes": len(local),
                "sha256": hashlib.sha256(local).hexdigest(),
            }
        )
    return {
        "path": str(root),
        "commit": commit,
        "requiredFilesMatchCommit": True,
        "requiredFiles": evidence,
    }


def _install_wespeaker_namespaces(source_path: Path) -> None:
    package_root = source_path / "wespeaker"
    for name, path in (
        ("wespeaker", package_root),
        ("wespeaker.models", package_root / "models"),
    ):
        existing = sys.modules.get(name)
        if existing is not None:
            existing_paths = tuple(getattr(existing, "__path__", ()))
            if str(path) not in existing_paths:
                raise W2VBert2BenchmarkError(
                    f"conflicting Python module is already loaded: {name}"
                )
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _inference_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw)
    if (
        config.get("model_type") != "wav2vec2-bert"
        or config.get("hidden_size") != 1024
        or config.get("num_hidden_layers") != 24
    ):
        raise W2VBert2BenchmarkError("W2V-BERT2 base config is unsupported")
    config.update(
        {
            "apply_spec_augment": False,
            "mask_feature_prob": 0.0,
            "mask_time_prob": 0.0,
        }
    )
    return config


def _checkpoint_without_projection(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    state = dict(checkpoint)
    present = {key for key in state if key.startswith("projection.")}
    if present != EXCLUDED_CLASSIFICATION_KEYS:
        raise W2VBert2BenchmarkError(
            "checkpoint classification keys are unexpected"
        )
    for key in EXCLUDED_CLASSIFICATION_KEYS:
        state.pop(key)
    return state


def _load_model(
    model_path: Path,
    wespeaker_source: Path,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        import yaml
        from transformers import (
            AutoFeatureExtractor,
            Wav2Vec2BertConfig,
            Wav2Vec2BertModel,
        )
    except ImportError as exc:
        raise W2VBert2BenchmarkError(
            "PyTorch, PyYAML, and Transformers are required"
        ) from exc

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise W2VBert2BenchmarkError("CUDA was requested but is unavailable")
    config_yaml = yaml.safe_load(
        (model_path / "config_LM.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(config_yaml, dict):
        raise W2VBert2BenchmarkError("candidate YAML is invalid")
    expected_args = {
        "feat_dim": 1024,
        "embed_dim": 256,
        "pooling_func": "ASP",
        "n_mfa_layers": -1,
        "adapter_dim": 128,
        "dropout": 0.0,
        "num_frontend_hidden_layers": 24,
    }
    if (
        config_yaml.get("model") != "W2VBert_Adapter_MFA"
        or config_yaml.get("model_args") != expected_args
        or config_yaml.get("dataset_args", {}).get("resample_rate") != 16_000
    ):
        raise W2VBert2BenchmarkError("candidate architecture config mismatches")

    raw_base = _strict_object(model_path / "base-config" / "config.json")
    runtime_config = _inference_config(raw_base)
    _install_wespeaker_namespaces(wespeaker_source)
    adapter_module = importlib.import_module(
        "wespeaker.models.w2vbert_adapter_mfa"
    )
    model = adapter_module.W2VBert_Adapter_MFA(**expected_args)
    frontend = torch.nn.Module()
    frontend.add_module(
        "encoder",
        Wav2Vec2BertModel(Wav2Vec2BertConfig(**runtime_config)),
    )
    model.add_module("frontend", frontend)
    raw_checkpoint = torch.load(
        model_path / "w2v_bert2_voxceleb_reproduced_LM.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw_checkpoint, Mapping):
        raise W2VBert2BenchmarkError("candidate checkpoint is not a state dict")
    checkpoint = _checkpoint_without_projection(raw_checkpoint)
    try:
        incompatible = model.load_state_dict(checkpoint, strict=True, assign=True)
    except RuntimeError as exc:
        raise W2VBert2BenchmarkError("candidate checkpoint is incompatible") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise W2VBert2BenchmarkError("candidate checkpoint load is incomplete")
    extractor = AutoFeatureExtractor.from_pretrained(
        model_path / "base-config",
        local_files_only=True,
    )
    return model.to(target_device).eval(), extractor, runtime_config


def _process_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def run_benchmark(
    *,
    model_path: Path,
    candidate_lock_path: Path,
    wespeaker_source: Path,
    trial_manifest_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise W2VBert2BenchmarkError("batch size must be positive")
    model_evidence = verify_candidate_snapshot(model_path, candidate_lock_path)
    source_evidence = verify_wespeaker_source(wespeaker_source)
    manifest = load_trial_manifest(trial_manifest_path)
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise W2VBert2BenchmarkError(
            "NumPy, SoundFile, and PyTorch are required"
        ) from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise W2VBert2BenchmarkError("CUDA benchmark requested but unavailable")

    torch.manual_seed(0)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    memory_before = _process_memory_bytes()
    load_started = time.perf_counter()
    model, extractor, runtime_config = _load_model(
        model_path.resolve(), wespeaker_source.resolve(), device
    )
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
        raise W2VBert2BenchmarkError("benchmark requires mono source audio")
    clip_rows = sorted(manifest["clips"], key=lambda row: str(row["clipId"]))
    clip_waves: list[Any] = []
    for clip in clip_rows:
        start_sample = round(int(clip["startMs"]) * sample_rate / 1000.0)
        end_sample = round(int(clip["endMs"]) * sample_rate / 1000.0)
        wave = np.asarray(audio[start_sample:end_sample, 0], dtype=np.float32)
        if wave.size < 1:
            raise W2VBert2BenchmarkError("trial clip is empty")
        clip_waves.append(wave)

    vectors: dict[str, Any] = {}
    embed_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(clip_rows), batch_size):
            rows = clip_rows[offset : offset + batch_size]
            waves = clip_waves[offset : offset + batch_size]
            features = extractor(
                waves,
                return_tensors="pt",
                sampling_rate=sample_rate,
                padding="longest",
            ).to(target_device)
            input_features = features.get("input_features")
            if input_features is None:
                raise W2VBert2BenchmarkError(
                    "feature extractor returned no input_features"
                )
            outputs = model.frontend.encoder(
                input_features,
                attention_mask=features.get("attention_mask"),
                output_hidden_states=True,
                return_dict=True,
            )
            if outputs.hidden_states is None or len(outputs.hidden_states) != 25:
                raise W2VBert2BenchmarkError(
                    "W2V-BERT2 hidden-state contract mismatches"
                )
            _, raw_embeddings = model(outputs.hidden_states)
            embeddings = torch.nn.functional.normalize(
                raw_embeddings.float(), dim=-1
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
        if not math.isfinite(score):
            raise W2VBert2BenchmarkError("verification score is non-finite")
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
    dimensions = {int(vector.shape[-1]) for vector in vectors.values()}
    if dimensions != {256}:
        raise W2VBert2BenchmarkError("embedding dimension contract mismatches")

    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "benchmark": "w2vbert2-frozen-speaker-verification",
        "promotionEligible": False,
        "promotionBlockers": [
            "research-only candidate license review",
            "development split cannot establish held-out non-regression",
        ],
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
            "transformers": importlib.import_module("transformers").__version__,
            "device": str(target_device),
            "deviceName": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else "cpu"
            ),
            "batchSize": batch_size,
            "excludedClassificationKeys": sorted(EXCLUDED_CLASSIFICATION_KEYS),
            "baseConfigOverrides": {
                key: runtime_config[key]
                for key in (
                    "apply_spec_augment",
                    "mask_feature_prob",
                    "mask_time_prob",
                )
            },
            "determinism": {
                "seed": 0,
                "cudnnDeterministic": (
                    bool(torch.backends.cudnn.deterministic)
                    if target_device.type == "cuda"
                    else None
                ),
                "tf32Allowed": False,
            },
        },
        "execution": {
            "speakerCount": manifest["counts"]["speakers"],
            "clipCount": len(clip_rows),
            "trialCount": len(trial_scores),
            "embeddingDimensions": 256,
            "modelLoadSeconds": load_seconds,
            "embeddingSeconds": embedding_seconds,
            "embeddingSecondsPerClip": embedding_seconds / len(clip_rows),
            "modelVramBytes": model_vram_bytes,
            "peakAllocatedVramBytes": peak_vram_bytes,
            "peakReservedVramBytes": peak_reserved_vram_bytes,
            "processRssBeforeBytes": memory_before,
            "processRssAfterBytes": memory_after,
        },
        "metrics": verification_metrics(metric_input),
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
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--wespeaker-source", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_integer, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        model_path=args.model_path,
        candidate_lock_path=args.candidate_lock,
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
