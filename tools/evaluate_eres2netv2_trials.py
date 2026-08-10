"""Evaluate one local ERes2NetV2 model on a frozen verification trial set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from backend.production_runners import (  # noqa: E402
    LocalERes2NetV2Verifier,
    _load_audio,
    _slice_audio,
)
from tools.benchmark_eres2netv2_residency import (  # noqa: E402
    _reset_cuda_peaks,
    _resource_snapshot,
    _validated_resource_snapshot,
)


ALLOWED_EVALUATION_SPLITS = frozenset(
    {"development", "regression", "held-out"}
)


class SpeakerVerificationEvaluationError(RuntimeError):
    """Raised when frozen evidence cannot support a valid evaluation."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeakerVerificationEvaluationError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpeakerVerificationEvaluationError(f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpeakerVerificationEvaluationError(
            f"{field} must be non-empty text"
        )
    return value.strip()


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpeakerVerificationEvaluationError(f"{field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field).casefold()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise SpeakerVerificationEvaluationError(f"{field} is not SHA-256")
    return text


def _local_file(value: Any, field: str) -> Path:
    text = _text(value, field)
    if "://" in text:
        raise SpeakerVerificationEvaluationError(f"{field} must be local")
    path = Path(text).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise SpeakerVerificationEvaluationError(
            f"{field} must be a regular local file"
        )
    return path


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpeakerVerificationEvaluationError(
            f"{field} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SpeakerVerificationEvaluationError(f"{field} must be an object")
    return value


def _verify_declared_file(
    path: Path,
    *,
    expected_bytes: Any,
    expected_sha256: Any,
    field: str,
) -> None:
    size = _integer(expected_bytes, f"{field}Bytes", minimum=1)
    digest = _sha256(expected_sha256, f"{field}Sha256")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise SpeakerVerificationEvaluationError(
            f"{field} no longer matches the frozen manifest"
        )


def _validate_annotation_clip(
    *,
    clip_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
    annotation: Mapping[str, Any],
) -> None:
    starts = _array(annotation.get("timestamps_start"), "annotation.timestamps_start")
    ends = _array(annotation.get("timestamps_end"), "annotation.timestamps_end")
    speakers = _array(annotation.get("speakers"), "annotation.speakers")
    if not starts or len(starts) != len(ends) or len(starts) != len(speakers):
        raise SpeakerVerificationEvaluationError(
            "annotation turn arrays are inconsistent"
        )
    target_intervals: list[tuple[int, int]] = []
    active_speakers: set[str] = set()
    for index, (raw_start, raw_end, raw_speaker) in enumerate(
        zip(starts, ends, speakers)
    ):
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, (int, float))
            or isinstance(raw_end, bool)
            or not isinstance(raw_end, (int, float))
        ):
            raise SpeakerVerificationEvaluationError(
                f"annotation turn {index} has invalid timing"
            )
        turn_start = round(float(raw_start) * 1000.0)
        turn_end = round(float(raw_end) * 1000.0)
        turn_speaker = _text(raw_speaker, f"annotation.speakers[{index}]")
        if turn_end <= start_ms or turn_start >= end_ms:
            continue
        active_speakers.add(turn_speaker)
        if turn_speaker == speaker_id:
            target_intervals.append(
                (max(start_ms, turn_start), min(end_ms, turn_end))
            )
    if active_speakers != {speaker_id}:
        raise SpeakerVerificationEvaluationError(
            f"clip {clip_id} is not single-active-speaker truth"
        )
    cursor = start_ms
    for interval_start, interval_end in sorted(target_intervals):
        if interval_start > cursor:
            break
        cursor = max(cursor, interval_end)
        if cursor >= end_ms:
            return
    raise SpeakerVerificationEvaluationError(
        f"clip {clip_id} is not fully covered by its declared speaker"
    )


def _load_trial_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    document = _load_json_object(resolved, "trial manifest")
    declared_canonical = _sha256(
        document.get("canonicalSha256"),
        "trial manifest canonicalSha256",
    )
    canonical_input = dict(document)
    canonical_input.pop("canonicalSha256", None)
    if canonical_json_sha256(canonical_input) != declared_canonical:
        raise SpeakerVerificationEvaluationError(
            "trial manifest canonicalSha256 does not match"
        )
    if document.get("schemaVersion") != "1.0.0":
        raise SpeakerVerificationEvaluationError(
            "trial manifest schemaVersion is unsupported"
        )

    source = _object(document.get("source"), "source")
    audio_path = _local_file(source.get("audioPath"), "source.audioPath")
    annotation_path = _local_file(
        source.get("annotationPath"), "source.annotationPath"
    )
    _verify_declared_file(
        audio_path,
        expected_bytes=source.get("audioBytes"),
        expected_sha256=source.get("audioSha256"),
        field="source.audio",
    )
    _verify_declared_file(
        annotation_path,
        expected_bytes=source.get("annotationBytes"),
        expected_sha256=source.get("annotationSha256"),
        field="source.annotation",
    )
    annotation = _load_json_object(annotation_path, "source annotation")
    recording_id = _text(source.get("recordingId"), "source.recordingId")

    raw_clips = _array(document.get("clips"), "clips")
    raw_trials = _array(document.get("trials"), "trials")
    if not raw_clips or not raw_trials:
        raise SpeakerVerificationEvaluationError(
            "trial manifest must contain clips and trials"
        )
    clips: dict[str, dict[str, Any]] = {}
    evaluation_splits: set[str] = set()
    for index, value in enumerate(raw_clips):
        clip = _object(value, f"clips[{index}]")
        clip_id = _text(clip.get("clipId"), f"clips[{index}].clipId")
        speaker_id = _text(
            clip.get("speakerId"), f"clips[{index}].speakerId"
        )
        start_ms = _integer(
            clip.get("startMs"), f"clips[{index}].startMs"
        )
        end_ms = _integer(
            clip.get("endMs"), f"clips[{index}].endMs", minimum=1
        )
        evaluation_split = _text(
            clip.get("evaluationSplit"),
            f"clips[{index}].evaluationSplit",
        )
        if evaluation_split not in ALLOWED_EVALUATION_SPLITS:
            raise SpeakerVerificationEvaluationError(
                f"clip {clip_id} has an invalid evaluation split"
            )
        if end_ms <= start_ms:
            raise SpeakerVerificationEvaluationError(
                f"clip {clip_id} has invalid boundaries"
            )
        if _local_file(clip.get("audio"), f"clips[{index}].audio") != audio_path:
            raise SpeakerVerificationEvaluationError(
                f"clip {clip_id} is rebound to another audio file"
            )
        if clip.get("sourceRecordingId") != recording_id:
            raise SpeakerVerificationEvaluationError(
                f"clip {clip_id} crosses recording boundaries"
            )
        if clip_id in clips:
            raise SpeakerVerificationEvaluationError(
                f"duplicate clip id: {clip_id}"
            )
        _validate_annotation_clip(
            clip_id=clip_id,
            speaker_id=speaker_id,
            start_ms=start_ms,
            end_ms=end_ms,
            annotation=annotation,
        )
        clips[clip_id] = {
            "clipId": clip_id,
            "speakerId": speaker_id,
            "startMs": start_ms,
            "endMs": end_ms,
            "evaluationSplit": evaluation_split,
        }
        evaluation_splits.add(evaluation_split)
    if len(evaluation_splits) != 1:
        raise SpeakerVerificationEvaluationError(
            "one evaluation report cannot mix evaluation splits"
        )
    evaluation_split = next(iter(evaluation_splits))

    trials: list[dict[str, Any]] = []
    trial_ids: set[str] = set()
    for index, value in enumerate(raw_trials):
        trial = _object(value, f"trials[{index}]")
        trial_id = _text(trial.get("trialId"), f"trials[{index}].trialId")
        enrollment_id = _text(
            trial.get("enrollmentClipId"),
            f"trials[{index}].enrollmentClipId",
        )
        test_id = _text(
            trial.get("testClipId"), f"trials[{index}].testClipId"
        )
        same_speaker = trial.get("sameSpeaker")
        if not isinstance(same_speaker, bool):
            raise SpeakerVerificationEvaluationError(
                f"trial {trial_id} sameSpeaker must be boolean"
            )
        if trial_id in trial_ids or enrollment_id == test_id:
            raise SpeakerVerificationEvaluationError(
                f"trial {trial_id} is duplicated or self-referential"
            )
        try:
            enrollment = clips[enrollment_id]
            test = clips[test_id]
        except KeyError as exc:
            raise SpeakerVerificationEvaluationError(
                f"trial {trial_id} references an unknown clip"
            ) from exc
        if (
            trial.get("evaluationSplit") != evaluation_split
            or enrollment["evaluationSplit"] != evaluation_split
            or test["evaluationSplit"] != evaluation_split
        ):
            raise SpeakerVerificationEvaluationError(
                f"trial {trial_id} crosses evaluation splits"
            )
        expected_same = enrollment["speakerId"] == test["speakerId"]
        if same_speaker != expected_same:
            raise SpeakerVerificationEvaluationError(
                f"trial {trial_id} has an incorrect truth label"
            )
        trial_ids.add(trial_id)
        trials.append(
            {
                "trialId": trial_id,
                "enrollmentClipId": enrollment_id,
                "testClipId": test_id,
                "sameSpeaker": same_speaker,
            }
        )

    counts = _object(document.get("counts"), "counts")
    same_count = sum(item["sameSpeaker"] for item in trials)
    different_count = len(trials) - same_count
    expected_counts = {
        "clips": len(clips),
        "speakers": len({item["speakerId"] for item in clips.values()}),
        "sameSpeakerTrials": same_count,
        "differentSpeakerTrials": different_count,
        "totalTrials": len(trials),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise SpeakerVerificationEvaluationError(
                f"counts.{key} does not match the frozen cases"
            )
    if same_count < 1 or different_count < 1:
        raise SpeakerVerificationEvaluationError(
            "both genuine and impostor trials are required"
        )
    return {
        "path": resolved,
        "fileSha256": sha256_file(resolved),
        "canonicalSha256": declared_canonical,
        "libraryId": _text(document.get("libraryId"), "libraryId"),
        "source": source,
        "audioPath": audio_path,
        "annotationPath": annotation_path,
        "recordingId": recording_id,
        "evaluationSplit": evaluation_split,
        "clips": tuple(clips.values()),
        "trials": tuple(trials),
    }


def _verified_model_evidence(model_path: Path) -> dict[str, Any]:
    root = model_path.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SpeakerVerificationEvaluationError(
            "model path must be a regular local directory"
        )
    manifest_path = root / ".mts-model-manifest.json"
    manifest = _load_json_object(manifest_path, "model manifest")
    files = _array(manifest.get("files"), "model manifest files")
    total = 0
    seen: set[str] = set()
    for index, value in enumerate(files):
        item = _object(value, f"model manifest files[{index}]")
        relative = _text(item.get("path"), f"model files[{index}].path")
        normalized = PurePosixPath(relative)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise SpeakerVerificationEvaluationError(
                "model manifest contains an unsafe path"
            )
        folded = normalized.as_posix().casefold()
        if folded in seen:
            raise SpeakerVerificationEvaluationError(
                "model manifest contains a duplicate path"
            )
        seen.add(folded)
        candidate = (root / Path(*normalized.parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SpeakerVerificationEvaluationError(
                "model file resolves outside the model directory"
            ) from exc
        size = _integer(
            item.get("size"), f"model files[{index}].size", minimum=1
        )
        digest = _sha256(
            item.get("sha256"), f"model files[{index}].sha256"
        )
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != size
            or sha256_file(candidate) != digest
        ):
            raise SpeakerVerificationEvaluationError(
                f"model file failed integrity verification: {relative}"
            )
        total += size
    if total != manifest.get("totalBytes"):
        raise SpeakerVerificationEvaluationError(
            "model manifest totalBytes does not match"
        )
    return {
        "path": str(root),
        "manifestPath": str(manifest_path),
        "manifestFileSha256": sha256_file(manifest_path),
        "manifestCanonicalSha256": canonical_json_sha256(manifest),
        "modelKey": _text(manifest.get("modelKey"), "model manifest modelKey"),
        "provider": _text(manifest.get("provider"), "model manifest provider"),
        "repoId": _text(manifest.get("repoId"), "model manifest repoId"),
        "revision": _text(manifest.get("revision"), "model manifest revision"),
        "lockSha256": _sha256(
            manifest.get("lockSha256"), "model manifest lockSha256"
        ),
        "totalBytes": total,
        "fileCount": len(files),
        "integrityVerified": True,
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise SpeakerVerificationEvaluationError(
            "embedding dimensions are inconsistent"
        )
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise SpeakerVerificationEvaluationError(
            "model returned a zero-norm embedding"
        )
    score = numerator / (left_norm * right_norm)
    if not math.isfinite(score):
        raise SpeakerVerificationEvaluationError(
            "model returned a non-finite trial score"
        )
    return max(-1.0, min(1.0, score))


def _embedding_set_sha256(vectors: Mapping[str, Sequence[float]]) -> str:
    """Hash clip-bound embeddings in a portable float32 representation."""

    digest = hashlib.sha256()
    for clip_id in sorted(vectors):
        encoded_id = clip_id.encode("utf-8")
        vector = tuple(float(value) for value in vectors[clip_id])
        if not vector or any(not math.isfinite(value) for value in vector):
            raise SpeakerVerificationEvaluationError(
                "embedding vector is invalid for hashing"
            )
        digest.update(len(encoded_id).to_bytes(4, "big"))
        digest.update(encoded_id)
        digest.update(len(vector).to_bytes(8, "big"))
        digest.update(struct.pack(f"<{len(vector)}f", *vector))
    if not vectors:
        raise SpeakerVerificationEvaluationError("embedding set is empty")
    return digest.hexdigest()


def _score_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise SpeakerVerificationEvaluationError("score class is empty")
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "populationStdDev": statistics.pstdev(values),
    }


def _operating_metrics(
    genuine: Sequence[float], impostor: Sequence[float]
) -> dict[str, float]:
    auc_numerator = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in genuine
        for negative in impostor
    )
    auc = auc_numerator / (len(genuine) * len(impostor))
    unique = sorted(set((*genuine, *impostor)))
    thresholds = [unique[0] - 1e-12]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    )
    thresholds.append(unique[-1] + 1e-12)
    rows: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        false_reject = sum(score < threshold for score in genuine) / len(genuine)
        false_accept = sum(score >= threshold for score in impostor) / len(impostor)
        rows.append((threshold, false_accept, false_reject))
    eer_row = min(
        rows,
        key=lambda item: (
            abs(item[1] - item[2]),
            item[1] + item[2],
            item[0],
        ),
    )
    min_dcf_row = min(
        rows,
        key=lambda item: (0.99 * item[1] + 0.01 * item[2], item[0]),
    )
    threshold, false_accept, false_reject = eer_row
    return {
        "rocAuc": auc,
        "equalErrorRate": (false_accept + false_reject) / 2.0,
        "equalErrorThreshold": threshold,
        "falseAcceptRateAtEerThreshold": false_accept,
        "falseRejectRateAtEerThreshold": false_reject,
        "balancedAccuracyAtEerThreshold": (
            1.0 - (false_accept + false_reject) / 2.0
        ),
        "minimumDetectionCostP01": (
            0.99 * min_dcf_row[1] + 0.01 * min_dcf_row[2]
        ),
        "minimumDetectionCostP01Threshold": min_dcf_row[0],
    }


def run_evaluation(
    *,
    model_path: Path,
    trial_manifest_path: Path,
    device: str,
    batch_size: int,
    verifier_factory: Callable[..., Any] = LocalERes2NetV2Verifier,
    resource_probe: Callable[[], Mapping[str, Any]] = _resource_snapshot,
    reset_resource_peaks: Callable[[], None] = _reset_cuda_peaks,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = _load_trial_manifest(trial_manifest_path)
    model = _verified_model_evidence(model_path)
    samples, sample_rate = _load_audio(manifest["audioPath"])
    source_audio = _object(manifest["source"].get("audio"), "source.audio")
    if (
        sample_rate != source_audio.get("sampleRateHz")
        or sample_rate != 16_000
        or round(len(samples) * 1000.0 / sample_rate)
        != source_audio.get("durationMs")
    ):
        raise SpeakerVerificationEvaluationError(
            "decoded source audio does not match the frozen description"
        )
    clips = manifest["clips"]
    audio_clips = [
        _slice_audio(samples, sample_rate, clip["startMs"], clip["endMs"])
        for clip in clips
    ]

    reset_resource_peaks()
    resource_before = _validated_resource_snapshot(resource_probe)
    verifier = verifier_factory(model_path=model_path, device=device)
    embeddings: list[tuple[float, ...]] = []
    started = time.perf_counter()
    try:
        for offset in range(0, len(audio_clips), batch_size):
            embeddings.extend(
                verifier._embeddings(audio_clips[offset : offset + batch_size])
            )
        inference_seconds = time.perf_counter() - started
        resource_after_inference = _validated_resource_snapshot(resource_probe)
    finally:
        verifier.release_resources()
        adapter_resources_released = verifier._pipeline_instance is None
    resource_after_release = _validated_resource_snapshot(resource_probe)
    if len(embeddings) != len(clips):
        raise SpeakerVerificationEvaluationError(
            "model did not return one embedding per frozen clip"
        )
    dimensions = {len(item) for item in embeddings}
    if len(dimensions) != 1 or next(iter(dimensions)) < 1:
        raise SpeakerVerificationEvaluationError(
            "model returned inconsistent embedding dimensions"
        )
    vectors = {
        clip["clipId"]: tuple(float(value) for value in embedding)
        for clip, embedding in zip(clips, embeddings)
    }
    scored_trials: list[dict[str, Any]] = []
    genuine: list[float] = []
    impostor: list[float] = []
    for trial in manifest["trials"]:
        score = _cosine(
            vectors[trial["enrollmentClipId"]],
            vectors[trial["testClipId"]],
        )
        (genuine if trial["sameSpeaker"] else impostor).append(score)
        scored_trials.append({**trial, "cosineScore": score})
    metrics = _operating_metrics(genuine, impostor)
    peak_rss = max(
        item["processRssMb"]
        for item in (
            resource_before,
            resource_after_inference,
            resource_after_release,
        )
    )
    peak_cuda_allocated = max(
        item["cudaPeakAllocatedMb"]
        for item in (
            resource_before,
            resource_after_inference,
            resource_after_release,
        )
    )
    peak_cuda_reserved = max(
        item["cudaPeakReservedMb"]
        for item in (
            resource_before,
            resource_after_inference,
            resource_after_release,
        )
    )
    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "benchmark": "frozen-speaker-verification-trials",
        "model": model,
        "trialManifest": {
            "path": str(manifest["path"]),
            "fileSha256": manifest["fileSha256"],
            "canonicalSha256": manifest["canonicalSha256"],
            "libraryId": manifest["libraryId"],
        },
        "partition": {
            "evaluationSplit": manifest["evaluationSplit"],
            "sourceRecordingId": manifest["recordingId"],
            "recordingCount": 1,
            "speakerCount": len({item["speakerId"] for item in clips}),
            "clipCount": len(clips),
            "trialCount": len(scored_trials),
            "genuineTrialCount": len(genuine),
            "impostorTrialCount": len(impostor),
        },
        "source": {
            "audioPath": str(manifest["audioPath"]),
            "audioSha256": manifest["source"]["audioSha256"],
            "annotationPath": str(manifest["annotationPath"]),
            "annotationSha256": manifest["source"]["annotationSha256"],
            "dataset": manifest["source"].get("dataset"),
            "revision": manifest["source"].get("revision"),
        },
        "execution": {
            "device": device,
            "batchSize": batch_size,
            "embeddingDimensions": next(iter(dimensions)),
            "embeddingSetHashAlgorithm": "clip-id-sorted-float32-le-v1",
            "embeddingSetSha256": _embedding_set_sha256(vectors),
            "inferenceSeconds": inference_seconds,
            "secondsPerClip": inference_seconds / len(clips),
            "adapterResourcesReleased": adapter_resources_released,
        },
        "resources": {
            "snapshots": {
                "beforeModelLoad": resource_before,
                "afterInference": resource_after_inference,
                "afterRelease": resource_after_release,
            },
            "peakProcessRssMb": peak_rss,
            "peakCudaAllocatedMb": peak_cuda_allocated,
            "peakCudaReservedMb": peak_cuda_reserved,
            "retainedCudaAllocatedMb": max(
                0.0,
                resource_after_release["cudaAllocatedMb"]
                - resource_before["cudaAllocatedMb"],
            ),
        },
        "scores": {
            "genuine": _score_summary(genuine),
            "impostor": _score_summary(impostor),
            **metrics,
        },
        "promotionPolicy": {
            "promotionEligibleBySplit": (
                manifest["evaluationSplit"] == "held-out"
            ),
            "decision": (
                "quality-comparison-eligible"
                if manifest["evaluationSplit"] == "held-out"
                else "development-or-regression-evidence-only"
            ),
        },
        "trials": scored_trials,
    }
    numeric_values = (
        inference_seconds,
        report["execution"]["secondsPerClip"],
        peak_rss,
        peak_cuda_allocated,
        peak_cuda_reserved,
        *metrics.values(),
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise SpeakerVerificationEvaluationError(
            "evaluation produced a non-finite metric"
        )
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite report: {resolved}")
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
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        model_path=args.model_path,
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
                "modelKey": report["model"]["modelKey"],
                "evaluationSplit": report["partition"]["evaluationSplit"],
                "scores": report["scores"],
                "execution": report["execution"],
                "resources": {
                    key: value
                    for key, value in report["resources"].items()
                    if key != "snapshots"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
