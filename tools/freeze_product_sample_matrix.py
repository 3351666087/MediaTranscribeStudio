"""Freeze all locally usable product samples into one truth-redacted matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    sha256_file,
)


DEFAULT_REQUIREMENTS = (
    PROJECT_ROOT / "sample_library" / "product-matrix-requirements.v1.json"
)
DEFAULT_GLOBAL_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_GLOBAL_AUDIO_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global" / "audio"
)
DEFAULT_CODE_SWITCH_MANIFEST = (
    PROJECT_ROOT / "sample_library" / "code-switch-manifest.v1.json"
)
DEFAULT_VOICE_ACTIVITY_MANIFEST = (
    PROJECT_ROOT / "sample_library" / "voice-activity-manifest.v1.json"
)


def _default_eval_root() -> Path:
    configured = os.environ.get("MTS_EVAL_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path("D:/mts-eval")
    return Path("/mnt/d/mts-eval")


DEFAULT_EVAL_ROOT = _default_eval_root()
DEFAULT_CODE_SWITCH_RESOLVED = (
    DEFAULT_EVAL_ROOT
    / "code-switch-v1"
    / "code-switch-sample-library.resolved.v1.json"
)
DEFAULT_VOICE_ACTIVITY_RESOLVED = (
    DEFAULT_EVAL_ROOT
    / "voice-activity-v1"
    / "voice-activity-samples.resolved.v1.json"
)
DEFAULT_REAL_DIARIZATION_MANIFESTS = (
    DEFAULT_EVAL_ROOT
    / "real-diarization-v1"
    / "global-real-diarization.resolved.v1.json",
    DEFAULT_EVAL_ROOT
    / "alimeeting-real-diarization-v1"
    / "global-real-diarization.resolved.v1.json",
)
DEFAULT_OUTPUT = (
    DEFAULT_EVAL_ROOT
    / "product-matrix-v1"
    / "truth-redacted-product-matrix.v1.json"
)
DEFAULT_SUPPORTING_ARTIFACTS = (
    (
        "aishell4-source-archive",
        DEFAULT_EVAL_ROOT / "source-archives" / "AISHELL-4" / "test.tar.gz",
        "source-only-not-product-case",
    ),
    (
        "aishell4-extracted-test-corpus",
        DEFAULT_EVAL_ROOT / "source-corpora" / "AISHELL-4" / "test",
        "source-only-not-product-case",
    ),
    (
        "alimeeting-eval-archive",
        DEFAULT_EVAL_ROOT / "source-archives" / "AliMeeting" / "Eval_Ali.tar.gz",
        "source-bound-by-resolved-cases",
    ),
)

_SPLITS = frozenset({"development", "regression", "held-out"})
_GLOBAL_TRUTH_FIELDS = frozenset(
    {
        "expectedTranscript",
        "scoringTranscript",
        "rawTranscript",
        "nativeTranscript",
        "englishTranscript",
        "referenceTranscript",
        "turns",
        "speakerSet",
        "overlapIntervals",
    }
)


class ProductMatrixFreezeError(ValueError):
    """Raised when a product matrix cannot be frozen without ambiguity."""


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductMatrixFreezeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProductMatrixFreezeError(f"{label} must contain an object")
    return value


def _source_id(value: Mapping[str, Any]) -> str | None:
    raw = value.get("id", value.get("sourceId"))
    return raw if isinstance(raw, str) and raw else None


def _source_index(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("sources")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        identifier = _source_id(row)
        if identifier is not None:
            result[identifier] = row
    return result


def _relative_path(path: Path, output_directory: Path) -> str:
    return Path(os.path.relpath(path, output_directory)).as_posix()


def _resolve_media_path(manifest_path: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    if os.name != "nt" and "\\" in raw_path:
        pure = PureWindowsPath(raw_path)
        if pure.drive:
            drive = pure.drive.rstrip(":").casefold()
            candidate = Path("/mnt") / drive / Path(*pure.parts[1:])
        else:
            candidate = Path(*pure.parts)
    else:
        candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _wav_duration(path: Path) -> float | None:
    if path.suffix.casefold() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise ProductMatrixFreezeError(f"invalid WAV media: {path}") from exc
    if frame_rate <= 0 or frame_count <= 0:
        raise ProductMatrixFreezeError(f"empty WAV media: {path}")
    return round(frame_count / frame_rate, 6)


def _declared_duration(row: Mapping[str, Any]) -> float | None:
    candidates: list[object] = [row.get("durationSeconds")]
    audio = row.get("audio")
    if isinstance(audio, dict):
        candidates.append(audio.get("durationSeconds"))
    for candidate in candidates:
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and float(candidate) > 0
        ):
            return round(float(candidate), 6)
    return None


def _media_evidence(
    path: Path,
    row: Mapping[str, Any],
    *,
    output_directory: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    declared_bytes = row.get("bytes")
    if (
        isinstance(declared_bytes, int)
        and not isinstance(declared_bytes, bool)
        and declared_bytes != actual_bytes
    ):
        raise ProductMatrixFreezeError(f"media byte count mismatch: {path}")
    actual_sha256 = sha256_file(path)
    declared_sha256 = row.get("sha256")
    if isinstance(declared_sha256, str) and declared_sha256 != actual_sha256:
        raise ProductMatrixFreezeError(f"media SHA-256 mismatch: {path}")
    actual_duration = _wav_duration(path)
    declared_duration = _declared_duration(row)
    if (
        actual_duration is not None
        and declared_duration is not None
        and abs(actual_duration - declared_duration) > 0.05
    ):
        raise ProductMatrixFreezeError(f"media duration mismatch: {path}")
    duration = actual_duration if actual_duration is not None else declared_duration
    if duration is None:
        raise ProductMatrixFreezeError(f"media duration is unavailable: {path}")
    return {
        "path": _relative_path(path, output_directory),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "durationSeconds": duration,
    }


def _recording_type(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    source_kind: str,
) -> str:
    declared = row.get("realOrSynthetic", source.get("recordingType"))
    if isinstance(declared, str) and declared:
        return declared
    scenarios = row.get("scenario")
    tags = set(scenarios) if isinstance(scenarios, list) else set()
    if "synthetic-mixture" in tags:
        return "synthetic-mixture"
    if "real-recording" in tags:
        return "real-recording"
    if source_kind == "voice-activity":
        return (
            "generated-fixture"
            if source.get("kind") == "generated"
            else "real-recording"
        )
    return "unknown"


def _language_tags(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    source_id: str,
    source_language_defaults: Mapping[str, Any],
) -> list[str]:
    expected = row.get("expectedLanguages")
    if isinstance(expected, list) and all(
        isinstance(item, str) and item for item in expected
    ):
        return list(dict.fromkeys(expected))
    language = row.get("language")
    if isinstance(language, str) and language not in {"", "auto", "mul"}:
        return [language]
    source_languages = source.get("languageTags")
    if isinstance(source_languages, list) and all(
        isinstance(item, str) and item for item in source_languages
    ):
        return list(dict.fromkeys(source_languages))
    fallback = source_language_defaults.get(source_id)
    if isinstance(fallback, list) and all(
        isinstance(item, str) and item for item in fallback
    ):
        return list(dict.fromkeys(fallback))
    return []


def _expected_speaker_count(row: Mapping[str, Any]) -> int | None:
    declared = row.get("expectedSpeakerCount")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared
    scenarios = row.get("scenario")
    if isinstance(scenarios, list) and "single-speaker" in scenarios:
        return 1
    return None


def _truth_availability(
    row: Mapping[str, Any],
    *,
    source_kind: str,
    source_id: str,
) -> dict[str, bool]:
    eligibility = row.get("truthEligibility")
    values = eligibility if isinstance(eligibility, dict) else {}
    scenarios = row.get("scenario")
    single_speaker = isinstance(scenarios, list) and "single-speaker" in scenarios
    transcript_sources = {
        "minds14",
        "fleurs",
        "librispeech",
        "mls",
        "ami-utterances",
    }
    return {
        "speakerCount": bool(values.get("speakerCount"))
        or (source_kind == "global" and single_speaker),
        "timeline": bool(values.get("turnBoundaries")),
        "overlap": bool(values.get("overlap")),
        "transcript": bool(values.get("asr"))
        or (source_kind == "global" and source_id in transcript_sources),
        "languageTiming": bool(values.get("languageTiming")),
        "voiceActivity": bool(values.get("voiceActivity")),
    }


def _safe_acquisition(row: Mapping[str, Any]) -> dict[str, Any] | None:
    acquisition = row.get("acquisition")
    if not isinstance(acquisition, dict):
        index = row.get("sourceRowIndex")
        return {"rowIndex": index} if isinstance(index, int) else None
    allowed = ("kind", "config", "split", "rowIndex", "rowIdField", "rowId")
    result = {key: acquisition[key] for key in allowed if key in acquisition}
    return result or None


def _recording_sha256(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> str | None:
    source_sha256 = source.get("sha256")
    if _is_sha256(source_sha256):
        return str(source_sha256)
    artifacts = row.get("sourceAudioArtifacts")
    if isinstance(artifacts, list) and len(artifacts) == 1:
        artifact = artifacts[0]
        if isinstance(artifact, Mapping) and _is_sha256(artifact.get("sha256")):
            return str(artifact["sha256"])
    return None


def _case_row(
    row: Mapping[str, Any],
    *,
    source_kind: str,
    source: Mapping[str, Any],
    manifest_sha256: str,
    manifest_path: Path,
    media_path: Path,
    output_directory: Path,
    evaluation_split: str,
    source_language_defaults: Mapping[str, Any],
    source_duration_seconds: float | None = None,
) -> dict[str, Any]:
    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ProductMatrixFreezeError("resolved case has no ID")
    source_identifier = row.get("sourceId")
    if not isinstance(source_identifier, str) or not source_identifier:
        raise ProductMatrixFreezeError(f"case has no sourceId: {case_id}")
    scenarios = row.get("scenario")
    scenario_tags = (
        list(dict.fromkeys(scenarios))
        if isinstance(scenarios, list)
        and all(isinstance(item, str) and item for item in scenarios)
        else []
    )
    overlap_intervals = row.get("overlapIntervals")
    if isinstance(overlap_intervals, list) and overlap_intervals:
        scenario_tags = list(dict.fromkeys([*scenario_tags, "overlap"]))
    provenance = {
        "sourceId": source_identifier,
        "dataset": source.get("dataset"),
        "revision": source.get("revision"),
        "license": source.get("license"),
        "sourceManifestSha256": manifest_sha256,
    }
    recording_sha256 = _recording_sha256(row, source)
    if recording_sha256 is not None:
        provenance["sourceMediaSha256"] = recording_sha256
    acquisition = _safe_acquisition(row)
    if acquisition is not None:
        provenance["acquisition"] = acquisition
    media = _media_evidence(
        media_path,
        row,
        output_directory=output_directory,
    )
    result: dict[str, Any] = {
        "id": case_id,
        "sourceKind": source_kind,
        "evaluationSplit": evaluation_split,
        "languageTags": _language_tags(
            row,
            source=source,
            source_id=source_identifier,
            source_language_defaults=source_language_defaults,
        ),
        "region": row.get("region"),
        "scenarios": scenario_tags,
        "signalClass": row.get("signalClass"),
        "recordingType": _recording_type(
            row,
            source,
            source_kind=source_kind,
        ),
        "expectedSpeakerCount": _expected_speaker_count(row),
        "media": media,
        "provenance": provenance,
        "truthAvailability": _truth_availability(
            row,
            source_kind=source_kind,
            source_id=source_identifier,
        ),
        "isolation": {
            "recordingIdentity": (
                "source-media-sha256"
                if recording_sha256 is not None
                else "media-sha256"
            ),
            "speakerIdentityAvailable": isinstance(row.get("speakerSet"), list),
            "speakerIdentityPersisted": False,
        },
    }
    if source_duration_seconds is not None:
        result["sourceDurationSeconds"] = round(source_duration_seconds, 6)
    license_evidence = source.get("licenseEvidence")
    if isinstance(license_evidence, dict):
        result["provenance"]["licenseEvidence"] = dict(license_evidence)
    return result


def _planned_split_index(global_manifest: Mapping[str, Any]) -> dict[str, str]:
    plans = global_manifest.get("plannedRealDiarizationSources")
    if not isinstance(plans, list):
        return {}
    result: dict[str, str] = {}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        source_id = plan.get("sourceId")
        split = plan.get("evaluationSplit")
        if isinstance(source_id, str) and split in _SPLITS:
            result[source_id] = str(split)
    return result


def _inventory_base(
    *,
    identifier: str,
    kind: str,
    path: Path,
    declared_cases: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "kind": kind,
        "path": str(path.resolve()),
        "state": "missing",
        "manifestSha256": None,
        "declaredCaseCount": declared_cases,
        "frozenCaseCount": 0,
        "missingCaseIds": [],
    }


def _ingest_global(
    *,
    manifest_path: Path,
    audio_root: Path,
    output_directory: Path,
    source_language_defaults: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    value = _load_object(manifest_path, label="global manifest")
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ProductMatrixFreezeError("global manifest has no cases")
    inventory = _inventory_base(
        identifier="global",
        kind="declared-audio",
        path=manifest_path,
        declared_cases=len(rows),
    )
    manifest_sha256 = sha256_file(manifest_path)
    inventory["manifestSha256"] = manifest_sha256
    sources = _source_index(value)
    frozen: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ProductMatrixFreezeError("global manifest case is not an object")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ProductMatrixFreezeError("global manifest case has no ID")
        forbidden = sorted(_GLOBAL_TRUTH_FIELDS.intersection(raw))
        if forbidden:
            raise ProductMatrixFreezeError(
                f"global case contains truth fields: {case_id}: {', '.join(forbidden)}"
            )
        split = raw.get("evaluationSplit")
        if split not in _SPLITS:
            raise ProductMatrixFreezeError(f"global case has invalid split: {case_id}")
        source_id = raw.get("sourceId")
        source = sources.get(str(source_id), {})
        media = audio_root / f"{case_id}.wav"
        if not media.is_file():
            missing.append(case_id)
            continue
        frozen.append(
            _case_row(
                raw,
                source_kind="global",
                source=source,
                manifest_sha256=manifest_sha256,
                manifest_path=manifest_path,
                media_path=media.resolve(),
                output_directory=output_directory,
                evaluation_split=str(split),
                source_language_defaults=source_language_defaults,
            )
        )
    inventory.update(
        {
            "state": "complete" if not missing else "partial",
            "frozenCaseCount": len(frozen),
            "missingCaseIds": sorted(missing),
        }
    )
    return frozen, inventory, value


def _declared_case_ids(path: Path, *, label: str) -> list[str]:
    value = _load_object(path, label=label)
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ProductMatrixFreezeError(f"{label} has no cases")
    result = [
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    if len(result) != len(rows) or len(set(result)) != len(result):
        raise ProductMatrixFreezeError(f"{label} case IDs are invalid")
    return result


def _source_duration_index(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    sources = value.get("sources")
    if not isinstance(sources, list):
        return result
    for source in sources:
        if not isinstance(source, dict):
            continue
        identifier = _source_id(source)
        if identifier is None:
            continue
        duration_ms = source.get("durationMs")
        duration_seconds = source.get("durationSeconds")
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
            result[identifier] = float(duration_ms) / 1000.0
        elif isinstance(duration_seconds, (int, float)) and not isinstance(
            duration_seconds, bool
        ):
            result[identifier] = float(duration_seconds)
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _validate_long_media_source(
    *,
    source_id: str,
    source: Mapping[str, Any],
    manifest_path: Path,
    source_durations: Mapping[str, float],
) -> None:
    """Require a locally rechecked, license-pinned original long recording."""

    source_sha256 = source.get("sha256")
    if not _is_sha256(source_sha256):
        raise ProductMatrixFreezeError(
            f"long-media source evidence SHA-256 is missing: {source_id}"
        )
    source_path = _resolve_media_path(manifest_path, source.get("path"))
    if source_path is None or not source_path.is_file():
        raise ProductMatrixFreezeError(
            f"long-media source evidence media is missing: {source_id}"
        )
    if sha256_file(source_path) != source_sha256:
        raise ProductMatrixFreezeError(
            f"long-media source evidence SHA-256 mismatch: {source_id}"
        )
    duration = source_durations.get(source_id)
    if duration is None or duration < 300.0:
        raise ProductMatrixFreezeError(
            f"long-media source duration is below 300 seconds: {source_id}"
        )
    if source.get("recordingType") != "real-recording":
        raise ProductMatrixFreezeError(
            f"long-media source is not a real recording: {source_id}"
        )
    for key in ("dataset", "revision", "license"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ProductMatrixFreezeError(
                f"long-media source evidence {key} is missing: {source_id}"
            )
    language_tags = source.get("languageTags")
    if not isinstance(language_tags, list) or not language_tags or any(
        not isinstance(tag, str) or not tag for tag in language_tags
    ):
        raise ProductMatrixFreezeError(
            f"long-media source language evidence is missing: {source_id}"
        )
    evidence = source.get("licenseEvidence")
    if not isinstance(evidence, dict):
        raise ProductMatrixFreezeError(
            f"long-media source evidence is missing: {source_id}"
        )
    for key in ("provider", "sourceUrl", "licenseUrl", "sourceReference"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ProductMatrixFreezeError(
                f"long-media source evidence {key} is missing: {source_id}"
            )
    if not evidence["sourceUrl"].startswith("https://") or not evidence[
        "licenseUrl"
    ].startswith("https://"):
        raise ProductMatrixFreezeError(
            f"long-media source evidence URLs are invalid: {source_id}"
        )
    if not _is_sha256(evidence.get("sourceReferenceSha256")):
        raise ProductMatrixFreezeError(
            f"long-media source reference SHA-256 is missing: {source_id}"
        )
    reference_path = _resolve_media_path(
        manifest_path,
        evidence.get("sourceReference"),
    )
    if reference_path is None or not reference_path.is_file():
        raise ProductMatrixFreezeError(
            f"long-media source reference is missing: {source_id}"
        )
    if sha256_file(reference_path) != evidence["sourceReferenceSha256"]:
        raise ProductMatrixFreezeError(
            f"long-media source reference SHA-256 mismatch: {source_id}"
        )
    if not (
        isinstance(evidence.get("pageRevisionId"), int)
        or isinstance(evidence.get("etag"), str)
    ):
        raise ProductMatrixFreezeError(
            f"long-media source immutable revision/ETag is missing: {source_id}"
        )
    try:
        reference_value = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductMatrixFreezeError(
            f"long-media source reference is invalid: {source_id}"
        ) from exc
    reference_source = (
        reference_value.get("source")
        if isinstance(reference_value, dict)
        else None
    )
    if not isinstance(reference_source, dict) or reference_source.get(
        "sha256"
    ) != source_sha256:
        raise ProductMatrixFreezeError(
            f"long-media source reference is not bound to media: {source_id}"
        )
    reference_license = reference_source.get("license")
    if not isinstance(reference_license, dict) or reference_license.get(
        "id"
    ) != source.get("license"):
        raise ProductMatrixFreezeError(
            f"long-media source reference license mismatch: {source_id}"
        )


def _ingest_resolved(
    *,
    identifier: str,
    source_kind: str,
    manifest_path: Path,
    output_directory: Path,
    source_language_defaults: Mapping[str, Any],
    source_evidence_overrides: Mapping[str, Mapping[str, Any]],
    planned_splits: Mapping[str, str],
    declared_case_ids: Sequence[str] = (),
    declaration_manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = _inventory_base(
        identifier=identifier,
        kind=source_kind,
        path=manifest_path,
        declared_cases=len(declared_case_ids),
    )
    if declaration_manifest_path is not None:
        inventory["declarationManifestPath"] = str(
            declaration_manifest_path.resolve()
        )
        inventory["declarationManifestSha256"] = sha256_file(
            declaration_manifest_path.resolve()
        )
    if not manifest_path.is_file():
        inventory["missingCaseIds"] = sorted(declared_case_ids)
        return [], inventory
    value = _load_object(manifest_path, label=f"{identifier} resolved manifest")
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ProductMatrixFreezeError(f"{identifier} resolved manifest has no cases")
    inventory["declaredCaseCount"] = max(len(rows), len(declared_case_ids))
    manifest_sha256 = sha256_file(manifest_path)
    inventory["manifestSha256"] = manifest_sha256
    sources = _source_index(value)
    source_durations = _source_duration_index(value)
    resolved_sources: dict[str, dict[str, Any]] = {}
    for source_id, raw_source in sources.items():
        source = dict(raw_source)
        source_sha256 = source.get("sha256")
        override = (
            source_evidence_overrides.get(source_sha256)
            if isinstance(source_sha256, str)
            else None
        )
        if isinstance(override, Mapping):
            source_path = _resolve_media_path(manifest_path, source.get("path"))
            if source_path is None or not source_path.is_file():
                raise ProductMatrixFreezeError(
                    f"source evidence media is missing: {source_id}"
                )
            if sha256_file(source_path) != source_sha256:
                raise ProductMatrixFreezeError(
                    f"source evidence SHA-256 mismatch: {source_id}"
                )
            source["license"] = override.get("license")
            source["languageTags"] = override.get("languageTags")
            source["licenseEvidence"] = {
                key: override[key]
                for key in ("provider", "pageId", "pageRevisionId", "sourceReference")
                if key in override
            }
        strict_long_media = source_kind == "long-media" and any(
            isinstance(row, dict)
            and row.get("evaluationSplit", planned_splits.get(str(row.get("sourceId"))))
            == "held-out"
            for row in rows
        )
        if strict_long_media:
            _validate_long_media_source(
                source_id=source_id,
                source=source,
                manifest_path=manifest_path,
                source_durations=source_durations,
            )
        resolved_sources[source_id] = source
    frozen: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ProductMatrixFreezeError(f"{identifier} case is not an object")
        case_id = raw.get("id")
        source_id = raw.get("sourceId")
        if not isinstance(case_id, str) or not isinstance(source_id, str):
            raise ProductMatrixFreezeError(f"{identifier} case identity is invalid")
        split = raw.get("evaluationSplit", planned_splits.get(source_id))
        if split not in _SPLITS:
            raise ProductMatrixFreezeError(
                f"{identifier} case has no valid evaluation split: {case_id}"
            )
        media = _resolve_media_path(manifest_path, raw.get("path"))
        if media is None or not media.is_file():
            missing.append(case_id)
            continue
        source = resolved_sources.get(source_id, {})
        frozen.append(
            _case_row(
                raw,
                source_kind=source_kind,
                source=source,
                manifest_sha256=manifest_sha256,
                manifest_path=manifest_path,
                media_path=media,
                output_directory=output_directory,
                evaluation_split=str(split),
                source_language_defaults=source_language_defaults,
                source_duration_seconds=source_durations.get(source_id),
            )
        )
    resolved_ids = {
        str(row.get("id")) for row in rows if isinstance(row, dict)
    }
    missing.extend(set(declared_case_ids) - resolved_ids)
    inventory.update(
        {
            "state": "complete" if not missing else "partial",
            "frozenCaseCount": len(frozen),
            "missingCaseIds": sorted(set(missing)),
        }
    )
    return frozen, inventory


def _case_qualifies(case: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    if policy.get("realRecordingRequired") is True and case.get(
        "recordingType"
    ) != "real-recording":
        return False
    if policy.get("licenseEvidenceRequired") is True:
        provenance = case.get("provenance")
        license_id = provenance.get("license") if isinstance(provenance, dict) else None
        if not isinstance(license_id, str) or not license_id:
            return False
    media = case.get("media")
    return (
        isinstance(media, dict)
        and isinstance(media.get("sha256"), str)
        and len(media["sha256"]) == 64
    )


def _language_match(case: Mapping[str, Any], bucket: Mapping[str, Any]) -> bool:
    languages = case.get("languageTags")
    tags = languages if isinstance(languages, list) else []
    exact = bucket.get("languageTags")
    if isinstance(exact, list) and set(tags).intersection(exact):
        return True
    prefixes = bucket.get("languagePrefixes")
    if isinstance(prefixes, list) and any(
        isinstance(tag, str)
        and any(tag.startswith(prefix) for prefix in prefixes if isinstance(prefix, str))
        for tag in tags
    ):
        return True
    minimum = bucket.get("minimumLanguageCount")
    scenarios = case.get("scenarios")
    scenario_tags = set(scenarios) if isinstance(scenarios, list) else set()
    expected_scenarios = bucket.get("scenarioTags")
    return (
        isinstance(minimum, int)
        and len(set(tags)) >= minimum
        and isinstance(expected_scenarios, list)
        and bool(scenario_tags.intersection(expected_scenarios))
    )


def _scenario_match(case: Mapping[str, Any], bucket: Mapping[str, Any]) -> bool:
    identifier = bucket.get("id")
    if identifier == "long-media":
        minimum = bucket.get("minimumSourceDurationSeconds")
        duration = case.get("sourceDurationSeconds")
        return (
            isinstance(minimum, (int, float))
            and not isinstance(minimum, bool)
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and float(duration) >= float(minimum)
        )
    scenarios = case.get("scenarios")
    scenario_tags = set(scenarios) if isinstance(scenarios, list) else set()
    expected_scenarios = bucket.get("scenarioTags")
    if isinstance(expected_scenarios, list) and scenario_tags.intersection(
        expected_scenarios
    ):
        return True
    signal_classes = bucket.get("signalClasses")
    return isinstance(signal_classes, list) and case.get("signalClass") in signal_classes


def _coverage_row(
    *,
    dimension: str,
    bucket: str,
    cases: Sequence[Mapping[str, Any]],
    required_splits: Sequence[str],
) -> dict[str, Any]:
    split_cases = {
        split: sorted(
            str(case["id"])
            for case in cases
            if case.get("evaluationSplit") == split
        )
        for split in required_splits
    }
    missing = [split for split, values in split_cases.items() if not values]
    def recording_identity(case: Mapping[str, Any]) -> str | None:
        provenance = case.get("provenance")
        if isinstance(provenance, dict) and _is_sha256(
            provenance.get("sourceMediaSha256")
        ):
            return str(provenance["sourceMediaSha256"])
        media = case.get("media")
        if isinstance(media, dict) and _is_sha256(media.get("sha256")):
            return str(media["sha256"])
        return None

    digests_by_split = {
        split: {
            identity
            for case in cases
            if case.get("evaluationSplit") == split
            for identity in (recording_identity(case),)
            if identity is not None
        }
        for split in required_splits
    }
    cross_split_duplicates: set[str] = set()
    for index, split in enumerate(required_splits):
        for other in required_splits[index + 1 :]:
            cross_split_duplicates.update(
                digests_by_split[split].intersection(digests_by_split[other])
            )
    truth_fields = (
        "speakerCount",
        "timeline",
        "overlap",
        "transcript",
        "languageTiming",
        "voiceActivity",
    )
    return {
        "dimension": dimension,
        "bucket": bucket,
        "complete": not missing and not cross_split_duplicates,
        "requiredSplitCaseCounts": {
            split: len(values) for split, values in split_cases.items()
        },
        "requiredSplitCaseIds": split_cases,
        "missingEvaluationSplits": missing,
        "licenseIds": sorted(
            {
                str(case["provenance"]["license"])
                for case in cases
                if isinstance(case.get("provenance"), dict)
                and isinstance(case["provenance"].get("license"), str)
            }
        ),
        "recordingIsolation": {
            "method": "source-media-sha256-when-available-else-media-sha256",
            "crossSplitDisjoint": not cross_split_duplicates,
            "duplicateSha256": sorted(cross_split_duplicates),
        },
        "speakerIsolation": {
            "status": "not-proven",
            "reason": "speaker identities are redacted or unavailable across corpora",
        },
        "referenceTruthAvailability": {
            field: any(
                isinstance(case.get("truthAvailability"), dict)
                and case["truthAvailability"].get(field) is True
                for case in cases
            )
            for field in truth_fields
        },
    }


def _build_coverage(
    *,
    requirements: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required_splits = requirements.get("requiredEvaluationSplits")
    if (
        not isinstance(required_splits, list)
        or set(required_splits) != {"development", "held-out"}
    ):
        raise ProductMatrixFreezeError(
            "requirements must declare development and held-out"
        )
    policy = requirements.get("qualificationPolicy")
    if not isinstance(policy, dict):
        raise ProductMatrixFreezeError("requirements qualificationPolicy is invalid")
    qualifying = [case for case in cases if _case_qualifies(case, policy)]
    coverage: list[dict[str, Any]] = []
    language_buckets = requirements.get("languageBuckets")
    if not isinstance(language_buckets, list):
        raise ProductMatrixFreezeError("requirements languageBuckets is invalid")
    for bucket in language_buckets:
        if not isinstance(bucket, dict) or not isinstance(bucket.get("id"), str):
            raise ProductMatrixFreezeError("language bucket is invalid")
        coverage.append(
            _coverage_row(
                dimension="language",
                bucket=str(bucket["id"]),
                cases=[case for case in qualifying if _language_match(case, bucket)],
                required_splits=required_splits,
            )
        )
    speaker_counts = requirements.get("speakerCountBuckets")
    if not isinstance(speaker_counts, list) or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 1
        for count in speaker_counts
    ):
        raise ProductMatrixFreezeError("requirements speakerCountBuckets is invalid")
    for count in speaker_counts:
        coverage.append(
            _coverage_row(
                dimension="speaker-count",
                bucket=f"N={count}",
                cases=[
                    case
                    for case in qualifying
                    if case.get("expectedSpeakerCount") == count
                ],
                required_splits=required_splits,
            )
        )
    scenario_buckets = requirements.get("scenarioBuckets")
    if not isinstance(scenario_buckets, list):
        raise ProductMatrixFreezeError("requirements scenarioBuckets is invalid")
    for bucket in scenario_buckets:
        if not isinstance(bucket, dict) or not isinstance(bucket.get("id"), str):
            raise ProductMatrixFreezeError("scenario bucket is invalid")
        coverage.append(
            _coverage_row(
                dimension="scenario",
                bucket=str(bucket["id"]),
                cases=[case for case in qualifying if _scenario_match(case, bucket)],
                required_splits=required_splits,
            )
        )
    gaps = []
    for row in coverage:
        if row["complete"]:
            continue
        if row["recordingIsolation"]["duplicateSha256"]:
            reason_code = "CROSS_SPLIT_RECORDING_LEAKAGE"
        elif sum(row["requiredSplitCaseCounts"].values()) == 0:
            reason_code = "NO_LOCAL_QUALIFYING_CASE"
        else:
            reason_code = "MISSING_REQUIRED_EVALUATION_SPLIT"
        gaps.append(
            {
            "id": f"{row['dimension']}:{row['bucket']}",
            "dimension": row["dimension"],
            "bucket": row["bucket"],
            "reasonCode": reason_code,
            "missingEvaluationSplits": row["missingEvaluationSplits"],
            "requiredSplitCaseCounts": row["requiredSplitCaseCounts"],
            }
        )
    return coverage, gaps


def _supporting_artifact(
    identifier: str,
    path: Path,
    role: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "id": identifier,
        "path": str(resolved),
        "role": role,
        "countsAsProductCase": False,
        "state": "missing",
    }
    if resolved.is_file():
        result.update(
            {
                "state": "present-file",
                "bytes": resolved.stat().st_size,
                "sha256": (
                    sha256_file(resolved)
                    if resolved.stat().st_size <= 64 * 1024 * 1024
                    else None
                ),
                "digestState": (
                    "verified"
                    if resolved.stat().st_size <= 64 * 1024 * 1024
                    else "not-recomputed-large-source-artifact"
                ),
            }
        )
    elif resolved.is_dir():
        files = [item for item in resolved.rglob("*") if item.is_file()]
        result.update(
            {
                "state": "present-directory",
                "fileCount": len(files),
                "bytes": sum(item.stat().st_size for item in files),
                "digestState": "not-recomputed-source-tree",
            }
        )
    return result


def build_product_matrix(
    *,
    requirements_path: Path,
    global_manifest_path: Path,
    global_audio_root: Path,
    code_switch_manifest_path: Path,
    code_switch_resolved_path: Path,
    voice_activity_manifest_path: Path,
    voice_activity_resolved_path: Path,
    real_diarization_manifest_paths: Sequence[Path],
    long_media_manifest_paths: Sequence[Path],
    supporting_artifacts: Sequence[tuple[str, Path, str]],
    output_path: Path,
) -> dict[str, Any]:
    requirements = _load_object(requirements_path, label="matrix requirements")
    if requirements.get("schemaVersion") != "1.0.0":
        raise ProductMatrixFreezeError("matrix requirements version is invalid")
    source_language_defaults = requirements.get("sourceLanguageDefaults")
    if not isinstance(source_language_defaults, dict):
        raise ProductMatrixFreezeError("sourceLanguageDefaults is invalid")
    raw_overrides = requirements.get("sourceEvidenceOverrides", [])
    if not isinstance(raw_overrides, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("sourceSha256"), str)
        or len(item["sourceSha256"]) != 64
        or not isinstance(item.get("license"), str)
        or not isinstance(item.get("languageTags"), list)
        for item in raw_overrides
    ):
        raise ProductMatrixFreezeError("sourceEvidenceOverrides is invalid")
    source_evidence_overrides = {
        str(item["sourceSha256"]): item for item in raw_overrides
    }
    if len(source_evidence_overrides) != len(raw_overrides):
        raise ProductMatrixFreezeError("sourceEvidenceOverrides contains duplicates")
    output_directory = output_path.resolve().parent

    cases, global_inventory, global_manifest = _ingest_global(
        manifest_path=global_manifest_path.resolve(),
        audio_root=global_audio_root.resolve(),
        output_directory=output_directory,
        source_language_defaults=source_language_defaults,
    )
    inventories = [global_inventory]
    planned_splits = _planned_split_index(global_manifest)

    code_switch_ids = _declared_case_ids(
        code_switch_manifest_path.resolve(),
        label="code-switch manifest",
    )
    resolved_cases, inventory = _ingest_resolved(
        identifier="code-switch",
        source_kind="code-switch",
        manifest_path=code_switch_resolved_path.resolve(),
        output_directory=output_directory,
        source_language_defaults=source_language_defaults,
        source_evidence_overrides=source_evidence_overrides,
        planned_splits=planned_splits,
        declared_case_ids=code_switch_ids,
        declaration_manifest_path=code_switch_manifest_path,
    )
    cases.extend(resolved_cases)
    inventories.append(inventory)

    voice_activity_ids = _declared_case_ids(
        voice_activity_manifest_path.resolve(),
        label="voice-activity manifest",
    )
    resolved_cases, inventory = _ingest_resolved(
        identifier="voice-activity",
        source_kind="voice-activity",
        manifest_path=voice_activity_resolved_path.resolve(),
        output_directory=output_directory,
        source_language_defaults=source_language_defaults,
        source_evidence_overrides=source_evidence_overrides,
        planned_splits=planned_splits,
        declared_case_ids=voice_activity_ids,
        declaration_manifest_path=voice_activity_manifest_path,
    )
    cases.extend(resolved_cases)
    inventories.append(inventory)

    for index, manifest_path in enumerate(real_diarization_manifest_paths, start=1):
        resolved_cases, inventory = _ingest_resolved(
            identifier=f"real-diarization-{index}",
            source_kind="real-diarization",
            manifest_path=manifest_path.resolve(),
            output_directory=output_directory,
            source_language_defaults=source_language_defaults,
            source_evidence_overrides=source_evidence_overrides,
            planned_splits=planned_splits,
        )
        cases.extend(resolved_cases)
        inventories.append(inventory)

    for index, manifest_path in enumerate(long_media_manifest_paths, start=1):
        resolved_cases, inventory = _ingest_resolved(
            identifier=f"long-media-{index}",
            source_kind="long-media",
            manifest_path=manifest_path.resolve(),
            output_directory=output_directory,
            source_language_defaults=source_language_defaults,
            source_evidence_overrides=source_evidence_overrides,
            planned_splits=planned_splits,
        )
        cases.extend(resolved_cases)
        inventories.append(inventory)

    case_ids = [str(case["id"]) for case in cases]
    duplicates = sorted(
        {case_id for case_id in case_ids if case_ids.count(case_id) > 1}
    )
    if duplicates:
        raise ProductMatrixFreezeError(
            "duplicate frozen case IDs: " + ", ".join(duplicates)
        )
    cases.sort(key=lambda case: str(case["id"]))
    coverage, gaps = _build_coverage(requirements=requirements, cases=cases)
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "truth-redacted-real-product-matrix",
        "matrixId": requirements.get("matrixId"),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "fileSha256": sha256_file(Path(__file__).resolve()),
        },
        "requirements": {
            "path": str(requirements_path.resolve()),
            "fileSha256": sha256_file(requirements_path.resolve()),
            "requiredEvaluationSplits": requirements["requiredEvaluationSplits"],
            "qualificationPolicy": requirements["qualificationPolicy"],
        },
        "truthPersistencePolicy": {
            "referenceTranscriptPersisted": False,
            "referenceTimelinePersisted": False,
            "speakerIdentityPersisted": False,
            "expectedSpeakerCountPersisted": True,
            "referenceAvailabilityFlagsPersisted": True,
        },
        "sourceInventory": inventories,
        "supportingArtifacts": [
            _supporting_artifact(identifier, path, role)
            for identifier, path, role in supporting_artifacts
        ],
        "counts": {
            "frozenCases": len(cases),
            "sourceInventories": len(inventories),
            "coverageBuckets": len(coverage),
            "completeCoverageBuckets": sum(row["complete"] for row in coverage),
            "gaps": len(gaps),
        },
        "completion": {
            "allRequiredBucketsComplete": not gaps,
            "taskChecklistMayBeMarkedComplete": False if gaps else True,
        },
        "coverage": coverage,
        "gaps": gaps,
        "cases": cases,
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--global-manifest", type=Path, default=DEFAULT_GLOBAL_MANIFEST)
    parser.add_argument("--global-audio-root", type=Path, default=DEFAULT_GLOBAL_AUDIO_ROOT)
    parser.add_argument(
        "--code-switch-manifest", type=Path, default=DEFAULT_CODE_SWITCH_MANIFEST
    )
    parser.add_argument(
        "--code-switch-resolved", type=Path, default=DEFAULT_CODE_SWITCH_RESOLVED
    )
    parser.add_argument(
        "--voice-activity-manifest",
        type=Path,
        default=DEFAULT_VOICE_ACTIVITY_MANIFEST,
    )
    parser.add_argument(
        "--voice-activity-resolved",
        type=Path,
        default=DEFAULT_VOICE_ACTIVITY_RESOLVED,
    )
    parser.add_argument("--real-diarization-manifest", action="append", type=Path)
    parser.add_argument("--long-media-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = build_product_matrix(
        requirements_path=args.requirements,
        global_manifest_path=args.global_manifest,
        global_audio_root=args.global_audio_root,
        code_switch_manifest_path=args.code_switch_manifest,
        code_switch_resolved_path=args.code_switch_resolved,
        voice_activity_manifest_path=args.voice_activity_manifest,
        voice_activity_resolved_path=args.voice_activity_resolved,
        real_diarization_manifest_paths=(
            args.real_diarization_manifest or DEFAULT_REAL_DIARIZATION_MANIFESTS
        ),
        long_media_manifest_paths=args.long_media_manifest,
        supporting_artifacts=DEFAULT_SUPPORTING_ARTIFACTS,
        output_path=args.output,
    )
    atomic_write_json_no_replace(args.output, matrix)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "caseCount": matrix["counts"]["frozenCases"],
                "completeCoverageBuckets": matrix["counts"][
                    "completeCoverageBuckets"
                ],
                "gapCount": matrix["counts"]["gaps"],
                "canonicalSha256": matrix["canonicalSha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
