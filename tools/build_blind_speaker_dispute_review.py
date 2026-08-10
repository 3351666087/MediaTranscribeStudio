"""Build a truth-redacted blind review package for speaker-verifier disputes.

Only development evidence is accepted. The reviewer packet contains cropped
WAV pairs and anonymous verifier opinions; trial truth, scores, thresholds,
speaker identities, model identities, source paths, and the ordering seed are
kept in a separate identity vault that is excluded from the reviewer ZIP.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import wave
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.errors import WorkerError  # noqa: E402
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


SCHEMA_VERSION = "1.0.0"
DEFAULT_CASE_COUNT = 36
DEFAULT_REVIEW_ROOT = "D:/mts-eval/blind-speaker-review"
DEFAULT_GATE_ROOT = "D:/mts-eval/gates/CN-Celeb-v2"
DEFAULT_DATASET_README = (
    "D:/mts-eval/datasets/CN-Celeb-v2/CN-Celeb_flac/README.TXT"
)
DEFAULT_LICENSE_LOCK = (
    PROJECT_ROOT / "benchmarks" / "speaker_models" / "cnceleb1-v2.dataset.lock.json"
)
_RELATION = {True: "same-person", False: "different-people"}
_SELECTION_BUCKETS = (
    "high-confidence-conflict",
    "prediction-disagreement",
    "threshold-near",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BlindSpeakerReviewError(ValueError):
    """Raised when evidence cannot be frozen without leakage or ambiguity."""


@dataclass(frozen=True)
class _Candidate:
    development_path: Path
    fixed_path: Path
    development: dict[str, Any]
    fixed: dict[str, Any]
    development_file_sha256: str
    fixed_file_sha256: str
    threshold: float
    scores: Mapping[str, float]


def _portable_path(value: str | Path, *, must_exist: bool = True) -> Path:
    raw = str(value)
    candidate = Path(raw).expanduser()
    if candidate.exists() or os.name == "nt":
        return candidate.resolve(strict=must_exist)
    windows = PureWindowsPath(raw)
    if windows.drive and len(windows.drive) == 2 and windows.drive[1] == ":":
        candidate = Path("/mnt") / windows.drive[0].lower()
        candidate = candidate.joinpath(*windows.parts[1:])
    return candidate.resolve(strict=must_exist)


def _load_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _portable_path(path)
    try:
        value = read_json_strict(resolved)
    except (OSError, UnicodeError, ValueError, WorkerError) as exc:
        raise BlindSpeakerReviewError(f"{label} is invalid: {resolved}") from exc
    declared = value.get("canonicalSha256")
    body = dict(value)
    body.pop("canonicalSha256", None)
    if not isinstance(declared, str) or canonical_json_sha256(body) != declared:
        raise BlindSpeakerReviewError(f"{label} canonical SHA-256 does not match")
    return value, resolved


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _portable_path(path)
    try:
        value = read_json_strict(resolved)
    except (OSError, UnicodeError, ValueError, WorkerError) as exc:
        raise BlindSpeakerReviewError(f"{label} is invalid: {resolved}") from exc
    return value, resolved


def _object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BlindSpeakerReviewError(f"{label} must be an object")
    return value


def _array(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise BlindSpeakerReviewError(f"{label} must be a non-empty array")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlindSpeakerReviewError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BlindSpeakerReviewError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlindSpeakerReviewError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise BlindSpeakerReviewError(f"{label} must be finite")
    return output


def _with_canonical(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "canonicalSha256": canonical_json_sha256(value)}


def _ordered(seed: bytes, namespace: str, values: Sequence[Any], key) -> list[Any]:
    def ordering_key(value: Any) -> bytes:
        message = f"{namespace}\0{key(value)}".encode("utf-8")
        return hmac.new(seed, message, hashlib.sha256).digest()

    return sorted(values, key=ordering_key)


def _percentile_rank(sorted_values: Sequence[float], value: float) -> float:
    return bisect.bisect_right(sorted_values, value) / len(sorted_values)


def _validate_development_report(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], Path, dict[str, float]]:
    report, resolved = _load_canonical_json(path, label=label)
    if (
        report.get("schemaVersion") != SCHEMA_VERSION
        or report.get("benchmark") != "frozen-speaker-verification-trials"
    ):
        raise BlindSpeakerReviewError(f"{label} has an unsupported report type")
    partition = _object(report.get("partition"), label=f"{label}.partition")
    if partition.get("evaluationSplit") != "development":
        raise BlindSpeakerReviewError(f"{label} must be development-only")
    rows = _array(report.get("trials"), label=f"{label}.trials")
    scores: dict[str, float] = {}
    identities: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(rows):
        trial = _object(raw, label=f"{label}.trials[{index}]")
        trial_id = _text(trial.get("trialId"), label=f"{label}.trialId")
        identity = (
            trial_id,
            trial.get("enrollmentClipId"),
            trial.get("testClipId"),
            trial.get("sameSpeaker"),
        )
        if identity in identities or not isinstance(trial.get("sameSpeaker"), bool):
            raise BlindSpeakerReviewError(f"{label} contains invalid trial identity")
        score = _finite(trial.get("cosineScore"), label=f"{label}.{trial_id}.score")
        if score < -1.0 or score > 1.0:
            raise BlindSpeakerReviewError(f"{label}.{trial_id} score is out of bounds")
        identities.add(identity)
        scores[trial_id] = score
    if partition.get("trialCount") != len(rows):
        raise BlindSpeakerReviewError(f"{label} trial count does not match")
    return report, resolved, scores


def _candidate_evidence(
    development_paths: Sequence[Path],
    fixed_paths: Sequence[Path],
) -> tuple[_Candidate, _Candidate]:
    if len(development_paths) != 2 or len(fixed_paths) != 2:
        raise BlindSpeakerReviewError("exactly two development and fixed reports are required")
    developments = [
        _validate_development_report(path, label=f"development report {index}")
        for index, path in enumerate(development_paths, start=1)
    ]
    fixed_reports = [
        (*_load_canonical_json(path, label=f"fixed report {index}"),)
        for index, path in enumerate(fixed_paths, start=1)
    ]
    by_development_sha = {
        sha256_file(resolved): (report, resolved, scores)
        for report, resolved, scores in developments
    }
    if len(by_development_sha) != 2:
        raise BlindSpeakerReviewError("development reports must be distinct")
    candidates: list[_Candidate] = []
    matched_development: set[str] = set()
    for fixed, fixed_resolved in fixed_reports:
        if (
            fixed.get("schemaVersion") != SCHEMA_VERSION
            or fixed.get("evaluation") != "development-frozen-speaker-threshold"
        ):
            raise BlindSpeakerReviewError("fixed report has an unsupported report type")
        policy = _object(fixed.get("thresholdPolicy"), label="thresholdPolicy")
        if (
            policy.get("sourceSplit") != "development"
            or policy.get("heldOutThresholdFittingPerformed") is not False
        ):
            raise BlindSpeakerReviewError("threshold was not frozen on development")
        threshold = _finite(policy.get("threshold"), label="thresholdPolicy.threshold")
        development_link = _object(fixed.get("development"), label="fixed.development")
        report_sha = _sha(
            development_link.get("reportFileSha256"),
            label="fixed.development.reportFileSha256",
        )
        linked = by_development_sha.get(report_sha)
        if linked is None or report_sha in matched_development:
            raise BlindSpeakerReviewError("fixed report does not uniquely bind a development report")
        report, resolved, scores = linked
        manifest_evidence = _object(report.get("trialManifest"), label="trialManifest")
        if (
            development_link.get("reportCanonicalSha256")
            != report.get("canonicalSha256")
            or development_link.get("trialManifestCanonicalSha256")
            != manifest_evidence.get("canonicalSha256")
        ):
            raise BlindSpeakerReviewError("fixed report development binding differs")
        report_model = _object(report.get("model"), label="development.model")
        fixed_model = _object(fixed.get("model"), label="fixed.model")
        identity_fields = ("modelKey", "repoId", "manifestFileSha256")
        if any(report_model.get(key) != fixed_model.get(key) for key in identity_fields):
            raise BlindSpeakerReviewError("fixed and development model identities differ")
        matched_development.add(report_sha)
        candidates.append(
            _Candidate(
                development_path=resolved,
                fixed_path=fixed_resolved,
                development=report,
                fixed=fixed,
                development_file_sha256=report_sha,
                fixed_file_sha256=sha256_file(fixed_resolved),
                threshold=threshold,
                scores=scores,
            )
        )
    if len(candidates) != 2:
        raise BlindSpeakerReviewError("both development reports require fixed thresholds")
    return candidates[0], candidates[1]


def _load_shared_manifest(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    evidence_rows = [
        _object(item.development.get("trialManifest"), label="trialManifest")
        for item in candidates
    ]
    identities = {
        (
            item.get("fileSha256"),
            item.get("canonicalSha256"),
            item.get("libraryId"),
        )
        for item in evidence_rows
    }
    if len(identities) != 1:
        raise BlindSpeakerReviewError("candidate reports do not share one trial manifest")
    path = _portable_path(_text(evidence_rows[0].get("path"), label="trialManifest.path"))
    if sha256_file(path) != evidence_rows[0].get("fileSha256"):
        raise BlindSpeakerReviewError("trial manifest file SHA-256 differs")
    manifest, resolved = _load_canonical_json(path, label="trial manifest")
    if manifest.get("canonicalSha256") != evidence_rows[0].get("canonicalSha256"):
        raise BlindSpeakerReviewError("trial manifest canonical SHA-256 differs")
    source = _object(manifest.get("source"), label="manifest.source")
    if source.get("split") != "development":
        raise BlindSpeakerReviewError("trial manifest must be development-only")
    clips_raw = _array(manifest.get("clips"), label="manifest.clips")
    trials_raw = _array(manifest.get("trials"), label="manifest.trials")
    clips: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(clips_raw):
        clip = dict(_object(raw, label=f"manifest.clips[{index}]"))
        clip_id = _text(clip.get("clipId"), label="clipId")
        if clip_id in clips or clip.get("evaluationSplit") != "development":
            raise BlindSpeakerReviewError("manifest contains duplicate or non-development clip")
        start = clip.get("startMs")
        end = clip.get("endMs")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise BlindSpeakerReviewError(f"clip {clip_id} has an invalid range")
        _sha(clip.get("sourceFileSha256"), label=f"clip {clip_id} source SHA-256")
        clips[clip_id] = clip
    trials: list[dict[str, Any]] = []
    trial_ids: set[str] = set()
    for index, raw in enumerate(trials_raw):
        trial = dict(_object(raw, label=f"manifest.trials[{index}]"))
        trial_id = _text(trial.get("trialId"), label="trialId")
        left = clips.get(trial.get("enrollmentClipId"))
        right = clips.get(trial.get("testClipId"))
        same = trial.get("sameSpeaker")
        if (
            trial_id in trial_ids
            or left is None
            or right is None
            or not isinstance(same, bool)
            or trial.get("evaluationSplit") != "development"
            or same != (left.get("speakerId") == right.get("speakerId"))
        ):
            raise BlindSpeakerReviewError(f"trial {trial_id} identity is invalid")
        trial_ids.add(trial_id)
        trials.append(trial)
    manifest_identity = tuple(
        (
            row["trialId"],
            row["enrollmentClipId"],
            row["testClipId"],
            row["sameSpeaker"],
        )
        for row in trials
    )
    for candidate in candidates:
        report_identity = tuple(
            (
                row.get("trialId"),
                row.get("enrollmentClipId"),
                row.get("testClipId"),
                row.get("sameSpeaker"),
            )
            for row in candidate.development["trials"]
        )
        report_source = _object(candidate.development.get("source"), label="report.source")
        if (
            report_identity != manifest_identity
            or report_source.get("audioSha256") != source.get("audioSha256")
            or report_source.get("annotationSha256") != source.get("annotationSha256")
        ):
            raise BlindSpeakerReviewError("candidate report differs from shared manifest")
    return {
        "document": manifest,
        "path": resolved,
        "source": source,
        "clips": clips,
        "trials": trials,
        "fileSha256": sha256_file(resolved),
    }


def _selection_rows(
    candidates: Sequence[_Candidate],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    absolute_distances = [
        sorted(abs(score - candidate.threshold) for score in candidate.scores.values())
        for candidate in candidates
    ]
    clips = manifest["clips"]
    rows: list[dict[str, Any]] = []
    for trial in manifest["trials"]:
        trial_id = trial["trialId"]
        candidate_rows = []
        for index, candidate in enumerate(candidates):
            score = candidate.scores[trial_id]
            margin = score - candidate.threshold
            candidate_rows.append(
                {
                    "candidateIndex": index,
                    "score": score,
                    "threshold": candidate.threshold,
                    "margin": margin,
                    "absoluteMargin": abs(margin),
                    "absoluteMarginPercentile": _percentile_rank(
                        absolute_distances[index], abs(margin)
                    ),
                    "prediction": margin >= 0.0,
                }
            )
        left = clips[trial["enrollmentClipId"]]
        right = clips[trial["testClipId"]]
        rows.append(
            {
                "trial": trial,
                "left": left,
                "right": right,
                "candidateRows": candidate_rows,
                "disagreement": (
                    candidate_rows[0]["prediction"]
                    != candidate_rows[1]["prediction"]
                ),
                "conflictStrength": min(
                    item["absoluteMarginPercentile"] for item in candidate_rows
                ),
                "thresholdNearRank": min(
                    item["absoluteMarginPercentile"] for item in candidate_rows
                ),
                "genrePair": "+".join(
                    sorted((str(left.get("genre")), str(right.get("genre"))))
                ),
                "speakerTuple": tuple(
                    sorted({str(left.get("speakerId")), str(right.get("speakerId"))})
                ),
            }
        )
    return rows


def _pick_diverse(
    pool: Sequence[dict[str, Any]],
    *,
    count: int,
    selected_ids: set[str],
    genre_counts: Counter[str],
    speaker_counts: Counter[str],
    seed: bytes,
    namespace: str,
    high_first: bool,
    metric_pool_multiplier: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    truth_targets = (count // 2, count // 2)
    for same_speaker, target in zip((True, False), truth_targets, strict=True):
        metric = "conflictStrength" if high_first else "thresholdNearRank"
        eligible = [
            row
            for row in pool
            if row["trial"]["trialId"] not in selected_ids
            and row["trial"]["sameSpeaker"] is same_speaker
        ]
        eligible.sort(
            key=lambda row: -row[metric] if high_first else row[metric]
        )
        if metric_pool_multiplier is not None:
            eligible = eligible[: target * metric_pool_multiplier]
        for offset in range(target):
            available = [
                row
                for row in eligible
                if row["trial"]["trialId"] not in selected_ids
            ]
            if not available:
                raise BlindSpeakerReviewError(
                    f"selection bucket {namespace} cannot preserve truth balance"
                )
            chosen = min(
                available,
                key=lambda row: (
                    genre_counts[row["genrePair"]],
                    sum(speaker_counts[item] for item in row["speakerTuple"]),
                    -row[metric] if high_first else row[metric],
                    hmac.new(
                        seed,
                        f"pick\0{namespace}\0{same_speaker}\0{offset}\0{row['trial']['trialId']}".encode(
                            "utf-8"
                        ),
                        hashlib.sha256,
                    ).digest(),
                ),
            )
            trial_id = chosen["trial"]["trialId"]
            selected_ids.add(trial_id)
            genre_counts[chosen["genrePair"]] += 1
            for speaker in chosen["speakerTuple"]:
                speaker_counts[speaker] += 1
            selected.append(chosen)
    return selected


def _select_cases(
    rows: Sequence[dict[str, Any]],
    *,
    case_count: int,
    seed: bytes,
) -> list[dict[str, Any]]:
    if case_count < 6 or case_count % 6 != 0:
        raise BlindSpeakerReviewError("case count must be a positive multiple of 6")
    disagreements = [row for row in rows if row["disagreement"]]
    per_bucket = case_count // 3
    if len(disagreements) < per_bucket * 2:
        raise BlindSpeakerReviewError("not enough model disagreements for requested package")
    selected_ids: set[str] = set()
    genre_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    high = _pick_diverse(
        disagreements,
        count=per_bucket,
        selected_ids=selected_ids,
        genre_counts=genre_counts,
        speaker_counts=speaker_counts,
        seed=seed,
        namespace="high-confidence-conflict",
        high_first=True,
        metric_pool_multiplier=2,
    )
    selected.extend({**row, "selectionBucket": _SELECTION_BUCKETS[0]} for row in high)
    remaining_disagreements = [
        row for row in disagreements if row["trial"]["trialId"] not in selected_ids
    ]
    general = _pick_diverse(
        remaining_disagreements,
        count=per_bucket,
        selected_ids=selected_ids,
        genre_counts=genre_counts,
        speaker_counts=speaker_counts,
        seed=seed,
        namespace="prediction-disagreement",
        high_first=True,
        metric_pool_multiplier=3,
    )
    selected.extend({**row, "selectionBucket": _SELECTION_BUCKETS[1]} for row in general)
    remaining = [row for row in rows if row["trial"]["trialId"] not in selected_ids]
    near = _pick_diverse(
        remaining,
        count=per_bucket,
        selected_ids=selected_ids,
        genre_counts=genre_counts,
        speaker_counts=speaker_counts,
        seed=seed,
        namespace="threshold-near",
        high_first=False,
        metric_pool_multiplier=3,
    )
    selected.extend({**row, "selectionBucket": _SELECTION_BUCKETS[2]} for row in near)
    return _ordered(
        seed,
        "case-order",
        selected,
        lambda row: row["trial"]["trialId"],
    )


def _load_source_audio(source: Mapping[str, Any]) -> dict[str, Any]:
    path = _portable_path(_text(source.get("audioPath"), label="source.audioPath"))
    expected_sha = _sha(source.get("audioSha256"), label="source.audioSha256")
    if sha256_file(path) != expected_sha:
        raise BlindSpeakerReviewError("source composite audio SHA-256 differs")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            frames = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise BlindSpeakerReviewError("source composite is not valid WAV") from exc
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 16_000
        or compression != "NONE"
        or len(frames) != frame_count * sample_width
    ):
        raise BlindSpeakerReviewError("source composite must be mono 16 kHz PCM16")
    annotation_path = _portable_path(
        _text(source.get("annotationPath"), label="source.annotationPath")
    )
    annotation_sha = _sha(
        source.get("annotationSha256"), label="source.annotationSha256"
    )
    if sha256_file(annotation_path) != annotation_sha:
        raise BlindSpeakerReviewError("source annotation SHA-256 differs")
    return {
        "path": path,
        "frames": frames,
        "frameCount": frame_count,
        "sampleRateHz": sample_rate,
        "sampleWidthBytes": sample_width,
        "channels": channels,
        "annotationPath": annotation_path,
    }


def _clip_frames(audio: Mapping[str, Any], clip: Mapping[str, Any]) -> bytes:
    rate = int(audio["sampleRateHz"])
    width = int(audio["sampleWidthBytes"])
    start_frame = int(clip["startMs"]) * rate // 1000
    end_frame = int(clip["endMs"]) * rate // 1000
    if start_frame < 0 or end_frame <= start_frame or end_frame > audio["frameCount"]:
        raise BlindSpeakerReviewError("selected clip falls outside source audio")
    return audio["frames"][start_frame * width : end_frame * width]


def _write_wave(path: Path, frames: bytes, *, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(frames)


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _license_evidence(lock_path: Path, readme_path: Path) -> dict[str, Any]:
    lock, lock_resolved = _load_json(lock_path, label="dataset license lock")
    readme_resolved = _portable_path(readme_path)
    try:
        readme = readme_resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BlindSpeakerReviewError("dataset README is unreadable") from exc
    license_row = _object(lock.get("license"), label="dataset lock license")
    noncommercial = "no commercial usage is permitted" in readme.casefold()
    return {
        "public": {
            "datasetLockFileSha256": sha256_file(lock_resolved),
            "datasetReadmeFileSha256": sha256_file(readme_resolved),
            "catalogDeclaredSpdx": license_row.get("spdx"),
            "bundledReadmeForbidsCommercialUse": noncommercial,
            "termsConflictRequiresLegalReview": bool(
                noncommercial and license_row.get("spdx") == "CC-BY-SA-4.0"
            ),
            "packageUse": "local-research-review-only-do-not-redistribute",
        },
        "private": {
            "datasetLockPath": str(lock_resolved),
            "datasetReadmePath": str(readme_resolved),
            "datasetKey": lock.get("datasetKey"),
            "catalogUrl": _object(lock.get("source"), label="dataset source").get(
                "catalogUrl"
            ),
            "license": dict(license_row),
            "bundledReadmeForbidsCommercialUse": noncommercial,
        },
    }


def _model_identity(candidate: _Candidate) -> dict[str, Any]:
    model = _object(candidate.development.get("model"), label="model")
    return {
        "modelKey": model.get("modelKey"),
        "provider": model.get("provider"),
        "repoId": model.get("repoId"),
        "revision": model.get("revision"),
        "path": model.get("path"),
        "manifestPath": model.get("manifestPath"),
        "manifestFileSha256": model.get("manifestFileSha256"),
        "manifestCanonicalSha256": model.get("manifestCanonicalSha256"),
    }


def _candidate_report_commitment(candidates: Sequence[_Candidate]) -> str:
    files = sorted(
        item
        for candidate in candidates
        for item in (
            candidate.development_file_sha256,
            candidate.fixed_file_sha256,
        )
    )
    return canonical_json_sha256({"reportFileSha256": files})


def _verify_original_clip(clip: Mapping[str, Any]) -> dict[str, Any]:
    path = _portable_path(_text(clip.get("sourceFilePath"), label="sourceFilePath"))
    expected = _sha(clip.get("sourceFileSha256"), label="sourceFileSha256")
    if sha256_file(path) != expected or path.stat().st_size != clip.get("sourceFileBytes"):
        raise BlindSpeakerReviewError("selected original clip evidence differs")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": expected,
    }


def _blindness_scan(
    reviewer_root: Path,
    *,
    candidates: Sequence[_Candidate],
    selected: Sequence[dict[str, Any]],
) -> None:
    forbidden: set[str] = set()
    for candidate in candidates:
        model = _model_identity(candidate)
        forbidden.update(
            str(model[key]).casefold()
            for key in ("modelKey", "repoId", "path", "manifestPath")
            if isinstance(model.get(key), str) and len(str(model[key])) >= 4
        )
        forbidden.update(
            {
                candidate.development_path.name.casefold(),
                candidate.fixed_path.name.casefold(),
            }
        )
    for row in selected:
        forbidden.add(str(row["trial"]["trialId"]).casefold())
        for clip in (row["left"], row["right"]):
            for key in ("clipId", "speakerId", "originalRecordingId", "sourceFilePath"):
                value = clip.get(key)
                if isinstance(value, str) and len(value) >= 4:
                    forbidden.add(value.casefold())
    for path in reviewer_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() in {".wav"}:
            continue
        text = path.read_text(encoding="utf-8").casefold()
        leak = next((value for value in forbidden if value in text), None)
        if leak is not None:
            raise BlindSpeakerReviewError(
                f"reviewer packet leaks sealed identity in {path.name}"
            )
        if any(
            token in text
            for token in (
                '"samespeaker"',
                '"cosinescore"',
                '"threshold"',
                '"speakerid"',
                '"trialid"',
            )
        ):
            raise BlindSpeakerReviewError(
                f"reviewer packet leaks sealed evidence field in {path.name}"
            )


def _write_reviewer_zip(reviewer_root: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in reviewer_root.rglob("*") if item.is_file()):
            relative = Path("reviewer-packet") / path.relative_to(reviewer_root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def build_review_package(
    *,
    development_report_paths: Sequence[Path],
    fixed_threshold_report_paths: Sequence[Path],
    output_root: Path,
    license_lock_path: Path = DEFAULT_LICENSE_LOCK,
    dataset_readme_path: Path = Path(DEFAULT_DATASET_README),
    case_count: int = DEFAULT_CASE_COUNT,
    random_seed: bytes | None = None,
) -> dict[str, Any]:
    """Freeze one development-only dispute batch and return its local paths."""

    seed = random_seed if random_seed is not None else secrets.token_bytes(32)
    if not isinstance(seed, bytes) or len(seed) < 32:
        raise BlindSpeakerReviewError("ordering seed must contain at least 256 bits")
    candidates = _candidate_evidence(
        development_report_paths,
        fixed_threshold_report_paths,
    )
    manifest = _load_shared_manifest(candidates)
    rows = _selection_rows(candidates, manifest)
    selected = _select_cases(rows, case_count=case_count, seed=seed)
    audio = _load_source_audio(manifest["source"])
    license_evidence = _license_evidence(license_lock_path, dataset_readme_path)

    package_token = hmac.new(seed, b"package-id", hashlib.sha256).hexdigest()[:16]
    package_id = f"blind-speaker-review-{package_token}"
    base = _portable_path(output_root, must_exist=False)
    base.mkdir(parents=True, exist_ok=True)
    final_root = base / package_id
    if final_root.exists():
        raise FileExistsError(f"refusing to replace blind review package: {final_root}")
    staging = Path(tempfile.mkdtemp(prefix=f".{package_id}.", dir=base))
    reviewer_root = staging / "reviewer-packet"
    vault_root = staging / "identity-vault"
    reviewer_root.mkdir()
    vault_root.mkdir()
    try:
        public_cases: list[dict[str, Any]] = []
        opinion_cases: list[dict[str, Any]] = []
        private_cases: list[dict[str, Any]] = []
        for order_index, row in enumerate(selected, start=1):
            trial_id = row["trial"]["trialId"]
            case_token = hmac.new(
                seed, f"case-id\0{trial_id}".encode("utf-8"), hashlib.sha256
            ).hexdigest()[:12]
            case_id = f"case-{case_token}"
            side_order = _ordered(
                seed,
                f"audio-side\0{trial_id}",
                (("enrollment", row["left"]), ("test", row["right"])),
                lambda item: item[0],
            )
            case_media = reviewer_root / "media" / case_id
            public_audio: list[dict[str, Any]] = []
            private_sides: list[dict[str, Any]] = []
            side_frames: list[bytes] = []
            for side_label, (source_role, clip) in zip(("A", "B"), side_order, strict=True):
                frames = _clip_frames(audio, clip)
                side_frames.append(frames)
                output = case_media / f"sample-{side_label.casefold()}.wav"
                _write_wave(output, frames)
                original = _verify_original_clip(clip)
                artifact = {
                    "side": side_label,
                    "path": output.relative_to(reviewer_root).as_posix(),
                    "durationMs": clip["endMs"] - clip["startMs"],
                    "bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                    "sourceClipCommitmentSha256": clip["sourceFileSha256"],
                }
                public_audio.append(artifact)
                private_sides.append(
                    {
                        **artifact,
                        "sourceRole": source_role,
                        "clipId": clip["clipId"],
                        "speakerId": clip["speakerId"],
                        "genre": clip.get("genre"),
                        "originalRecordingId": clip.get("originalRecordingId"),
                        "compositeStartMs": clip["startMs"],
                        "compositeEndMs": clip["endMs"],
                        "originalSource": original,
                    }
                )
            silence = b"\0\0" * (16_000 * 3 // 4)
            pair_path = case_media / "pair-a-then-b.wav"
            _write_wave(pair_path, side_frames[0] + silence + side_frames[1])
            pair_artifact = {
                "path": pair_path.relative_to(reviewer_root).as_posix(),
                "order": ["A", "B"],
                "separatorMs": 750,
                "durationMs": sum(item["durationMs"] for item in public_audio) + 750,
                "bytes": pair_path.stat().st_size,
                "sha256": sha256_file(pair_path),
            }
            candidate_order = _ordered(
                seed,
                f"candidate-order\0{trial_id}",
                list(range(len(candidates))),
                str,
            )
            opinions: list[dict[str, Any]] = []
            private_options: list[dict[str, Any]] = []
            for option_number, candidate_index in enumerate(candidate_order, start=1):
                option_id = f"option-{option_number:02d}"
                candidate_row = row["candidateRows"][candidate_index]
                opinions.append(
                    {
                        "optionId": option_id,
                        "relationship": _RELATION[candidate_row["prediction"]],
                    }
                )
                candidate = candidates[candidate_index]
                private_options.append(
                    {
                        "optionId": option_id,
                        "candidateIndex": candidate_index,
                        "model": _model_identity(candidate),
                        "developmentReportFileSha256": (
                            candidate.development_file_sha256
                        ),
                        "fixedReportFileSha256": candidate.fixed_file_sha256,
                        "cosineScore": candidate_row["score"],
                        "threshold": candidate_row["threshold"],
                        "signedMargin": candidate_row["margin"],
                        "absoluteMarginPercentile": candidate_row[
                            "absoluteMarginPercentile"
                        ],
                        "prediction": _RELATION[candidate_row["prediction"]],
                    }
                )
            public_cases.append(
                {
                    "caseId": case_id,
                    "presentationOrder": order_index,
                    "audio": public_audio,
                    "combinedPlayback": pair_artifact,
                }
            )
            opinion_cases.append({"caseId": case_id, "candidateOpinions": opinions})
            private_cases.append(
                {
                    "caseId": case_id,
                    "presentationOrder": order_index,
                    "selectionBucket": row["selectionBucket"],
                    "trialId": trial_id,
                    "sameSpeaker": row["trial"]["sameSpeaker"],
                    "truthRelationship": _RELATION[row["trial"]["sameSpeaker"]],
                    "genrePair": row["genrePair"],
                    "speakerTuple": list(row["speakerTuple"]),
                    "conflictStrength": row["conflictStrength"],
                    "thresholdNearRank": row["thresholdNearRank"],
                    "audioSideMapping": private_sides,
                    "candidateOptionMapping": private_options,
                }
            )

        report_set_commitment = _candidate_report_commitment(candidates)
        vault_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-speaker-review-identity-vault",
            "packageId": package_id,
            "confidentiality": "DO-NOT-SHARE-WITH-REVIEWER",
            "evaluationSplit": "development",
            "ordering": {
                "algorithm": "HMAC-SHA256-with-256-bit-secret-v1",
                "seedHex": seed.hex(),
                "caseIdsRandomized": True,
                "caseOrderRandomized": True,
                "audioSideOrderRandomizedPerCase": True,
                "candidateOrderRandomizedPerCase": True,
            },
            "selectionPolicy": {
                "caseCount": case_count,
                "buckets": list(_SELECTION_BUCKETS),
                "casesPerBucket": case_count // 3,
                "sameAndDifferentCasesPerBucket": case_count // 6,
                "highConfidenceDefinition": (
                    "largest minimum empirical percentile of absolute "
                    "development-threshold margin across both candidates"
                ),
                "thresholdNearDefinition": (
                    "smallest minimum empirical percentile of absolute "
                    "development-threshold margin across both candidates"
                ),
                "diversity": "minimize repeated genre pairs and speaker identities",
            },
            "sourceEvidence": {
                "trialManifestPath": str(manifest["path"]),
                "trialManifestFileSha256": manifest["fileSha256"],
                "trialManifestCanonicalSha256": manifest["document"][
                    "canonicalSha256"
                ],
                "compositeAudioPath": str(audio["path"]),
                "compositeAudioFileSha256": manifest["source"]["audioSha256"],
                "annotationPath": str(audio["annotationPath"]),
                "annotationFileSha256": manifest["source"]["annotationSha256"],
                "dataset": manifest["source"].get("dataset"),
                "revision": manifest["source"].get("revision"),
            },
            "licenseEvidence": license_evidence["private"],
            "modelReportSetCommitmentSha256": report_set_commitment,
            "candidates": [
                {
                    "candidateIndex": index,
                    "model": _model_identity(candidate),
                    "developmentReport": {
                        "path": str(candidate.development_path),
                        "fileSha256": candidate.development_file_sha256,
                        "canonicalSha256": candidate.development[
                            "canonicalSha256"
                        ],
                    },
                    "fixedThresholdReport": {
                        "path": str(candidate.fixed_path),
                        "fileSha256": candidate.fixed_file_sha256,
                        "canonicalSha256": candidate.fixed["canonicalSha256"],
                    },
                    "developmentThreshold": candidate.threshold,
                }
                for index, candidate in enumerate(candidates)
            ],
            "cases": private_cases,
        }
        vault = _with_canonical(vault_body)
        vault_path = vault_root / "DO-NOT-SHARE.identity-mapping.v1.json"
        atomic_write_json_no_replace(vault_path, vault)
        try:
            vault_path.chmod(0o600)
        except OSError:
            pass

        review_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-speaker-dispute-review-packet",
            "packageId": package_id,
            "evaluationSplit": "development",
            "reviewProtocol": {
                "phase1": "listen before opening phase-2 candidate opinions",
                "phase2": "compare the sealed independent judgment to anonymous opinions",
                "allowedRelationships": [
                    "same-person",
                    "different-people",
                    "uncertain",
                ],
            },
            "blindnessPolicy": {
                "modelIdentityIncluded": False,
                "speakerIdentityIncluded": False,
                "referenceTruthIncluded": False,
                "automaticScoresIncluded": False,
                "confidenceIncluded": False,
                "sourcePathsIncluded": False,
            },
            "evidenceCommitments": {
                "trialManifestFileSha256": manifest["fileSha256"],
                "compositeAudioFileSha256": manifest["source"]["audioSha256"],
                "sourceAnnotationFileSha256": manifest["source"][
                    "annotationSha256"
                ],
                "modelReportSetCommitmentSha256": report_set_commitment,
                "identityVaultFileSha256": sha256_file(vault_path),
            },
            "licenseEvidence": license_evidence["public"],
            "counts": {"cases": len(public_cases), "audioFilesPerCase": 3},
            "cases": public_cases,
        }
        review_manifest = _with_canonical(review_body)
        review_manifest_path = reviewer_root / "phase-1.review-manifest.v1.json"
        atomic_write_json_no_replace(review_manifest_path, review_manifest)
        opinions_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-speaker-anonymous-candidate-opinions",
            "packageId": package_id,
            "openOnlyAfterPhase1": True,
            "automaticScoresIncluded": False,
            "modelIdentityIncluded": False,
            "cases": opinion_cases,
        }
        opinions_path = reviewer_root / "phase-2.candidate-opinions.v1.json"
        atomic_write_json_no_replace(opinions_path, _with_canonical(opinions_body))
        decisions_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-speaker-human-review-decisions",
            "packageId": package_id,
            "reviewer": "",
            "reviewedAt": "",
            "automaticScoring": False,
            "cases": [
                {
                    "caseId": item["caseId"],
                    "independentRelationship": "",
                    "preferredOptionIds": [],
                    "audibility": "",
                    "severity": "",
                    "rationale": "",
                    "audibleEvidence": "",
                }
                for item in public_cases
            ],
        }
        decisions_path = reviewer_root / "review-decisions.template.v1.json"
        atomic_write_json_no_replace(decisions_path, decisions_body)
        _write_text_exclusive(
            reviewer_root / "REVIEW.md",
            "# Blind speaker review\n\n"
            "1. Open `phase-1.review-manifest.v1.json` and listen to A, B, and "
            "the combined A-then-B WAV for each case.\n"
            "2. Record `same-person`, `different-people`, or `uncertain` in the "
            "decision template before opening phase 2.\n"
            "3. Open `phase-2.candidate-opinions.v1.json`, then record any "
            "preferred anonymous option IDs and a concrete audible rationale.\n"
            "4. Do not request or inspect the identity vault until every case is "
            "frozen. This packet is restricted to local research review and must "
            "not be redistributed.\n",
        )
        _blindness_scan(reviewer_root, candidates=candidates, selected=selected)
        reviewer_zip = staging / f"{package_id}.reviewer-only.zip"
        _write_reviewer_zip(reviewer_root, reviewer_zip)
        package_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-speaker-review-package-index",
            "packageId": package_id,
            "reviewerPacket": {
                "relativePath": "reviewer-packet",
                "manifestFileSha256": sha256_file(review_manifest_path),
                "zipRelativePath": reviewer_zip.name,
                "zipFileSha256": sha256_file(reviewer_zip),
            },
            "identityVault": {
                "relativePath": "identity-vault/DO-NOT-SHARE.identity-mapping.v1.json",
                "fileSha256": sha256_file(vault_path),
                "shareWithReviewer": False,
            },
            "evaluationSplit": "development",
            "caseCount": case_count,
            "modelReportSetCommitmentSha256": report_set_commitment,
        }
        package_manifest_path = staging / "package-manifest.v1.json"
        atomic_write_json_no_replace(package_manifest_path, _with_canonical(package_body))
        os.rename(staging, final_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "packageId": package_id,
        "packageRoot": str(final_root),
        "reviewerPacket": str(final_root / "reviewer-packet"),
        "reviewerZip": str(final_root / f"{package_id}.reviewer-only.zip"),
        "identityMapping": str(
            final_root / "identity-vault" / "DO-NOT-SHARE.identity-mapping.v1.json"
        ),
        "packageManifest": str(final_root / "package-manifest.v1.json"),
        "caseCount": case_count,
    }


def _default_gate_file(name: str) -> Path:
    return Path(DEFAULT_GATE_ROOT) / "reports" / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report",
        action="append",
        type=Path,
        default=[],
        help="repeat exactly twice; defaults to CAM++ and ERes2NetV2-wide",
    )
    parser.add_argument(
        "--fixed-threshold-report",
        action="append",
        type=Path,
        default=[],
        help="repeat exactly twice; defaults to matching development thresholds",
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--license-lock", type=Path, default=DEFAULT_LICENSE_LOCK)
    parser.add_argument(
        "--dataset-readme", type=Path, default=Path(DEFAULT_DATASET_README)
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        help="optional private binary seed file with at least 32 bytes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    development = args.development_report or [
        _default_gate_file("campplus.development.v1.json"),
        _default_gate_file("eres2netv2-wide.development.v1.json"),
    ]
    fixed = args.fixed_threshold_report or [
        _default_gate_file("campplus.fixed-threshold.v1.json"),
        _default_gate_file("eres2netv2-wide.fixed-threshold.v1.json"),
    ]
    seed = None
    if args.seed_file is not None:
        seed = _portable_path(args.seed_file).read_bytes()
    result = build_review_package(
        development_report_paths=development,
        fixed_threshold_report_paths=fixed,
        output_root=args.output_root,
        license_lock_path=args.license_lock,
        dataset_readme_path=args.dataset_readme,
        case_count=args.case_count,
        random_seed=seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
