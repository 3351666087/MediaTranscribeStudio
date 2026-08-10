"""Freeze locally available media into a truth-redacted product matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    sha256_file,
)


DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_AUDIO_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global" / "audio"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / ".runtime_cache"
    / "sample-library"
    / "global"
    / "blind-product-development.v1.json"
)
_ALLOWED_SPLITS = ("development", "regression", "held-out")
_TRUTH_KEYS = {
    "expectedTranscript",
    "scoringTranscript",
    "rawTranscript",
    "nativeTranscript",
    "englishTranscript",
    "referenceTranscript",
    "rttm",
    "expectedTimeline",
}


class BlindProductMatrixError(ValueError):
    """Raised when local media cannot be frozen without ambiguity."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindProductMatrixError(f"invalid source manifest: {path}") from exc
    if not isinstance(value, dict):
        raise BlindProductMatrixError("source manifest must be an object")
    return value


def _wav_evidence(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise BlindProductMatrixError(f"invalid WAV media: {path}") from exc
    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise BlindProductMatrixError(
            f"WAV is not mono 16 kHz PCM16: {path.name}"
        )
    if frame_count < 1:
        raise BlindProductMatrixError(f"WAV contains no frames: {path.name}")
    return {
        "durationSeconds": round(frame_count / sample_rate, 6),
        "channels": channels,
        "sampleRateHz": sample_rate,
        "sampleWidthBytes": sample_width,
        "frameCount": frame_count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relative_media_path(media: Path, manifest_directory: Path) -> str:
    try:
        return media.relative_to(manifest_directory).as_posix()
    except ValueError:
        return Path(os.path.relpath(media, manifest_directory)).as_posix()


def _selected_rows(
    source: dict[str, Any],
    *,
    splits: Sequence[str],
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = source.get("cases")
    if not isinstance(rows, list) or not rows:
        raise BlindProductMatrixError("source manifest has no cases")
    available = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    unknown = sorted(set(case_ids) - set(available))
    if unknown:
        raise BlindProductMatrixError("unknown case IDs: " + ", ".join(unknown))
    selected_ids = set(case_ids)
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("evaluationSplit") in splits
        and (not selected_ids or row.get("id") in selected_ids)
    ]


def build_matrix(
    *,
    source_manifest: Path,
    audio_root: Path,
    output_path: Path,
    splits: Sequence[str] = ("development",),
    case_ids: Sequence[str] = (),
    allow_missing: bool = False,
    unlock_held_out: bool = False,
) -> dict[str, Any]:
    normalized_splits = tuple(dict.fromkeys(splits))
    if not normalized_splits or any(
        item not in _ALLOWED_SPLITS for item in normalized_splits
    ):
        raise BlindProductMatrixError("matrix split selection is invalid")
    if "held-out" in normalized_splits and not unlock_held_out:
        raise BlindProductMatrixError("held-out media requires explicit unlock")
    source_path = source_manifest.resolve(strict=True)
    media_root = audio_root.resolve(strict=True)
    source = _load_json_object(source_path)
    rows = _selected_rows(
        source,
        splits=normalized_splits,
        case_ids=case_ids,
    )
    if not rows:
        raise BlindProductMatrixError("matrix selection contains no cases")

    frozen_cases: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    for row in rows:
        case_id = str(row["id"])
        media = media_root / f"{case_id}.wav"
        if not media.is_file():
            missing_cases.append(case_id)
            continue
        if any(key in row for key in _TRUTH_KEYS):
            raise BlindProductMatrixError(
                f"source case unexpectedly contains truth field: {case_id}"
            )
        evidence = _wav_evidence(media)
        maximum = source.get("maxDurationSeconds")
        if isinstance(maximum, (int, float)) and evidence["durationSeconds"] > maximum:
            raise BlindProductMatrixError(f"media exceeds duration limit: {case_id}")
        frozen_cases.append(
            {
                "id": case_id,
                "sourceId": row.get("sourceId"),
                "language": row.get("language"),
                "region": row.get("region"),
                "evaluationSplit": row.get("evaluationSplit"),
                "scenario": row.get("scenario", []),
                "expectedSpeakerCount": row.get("expectedSpeakerCount"),
                "path": _relative_media_path(media, output_path.parent.resolve()),
                **evidence,
                "sourceLocator": {
                    "acquisition": row.get("acquisition"),
                    "sourceManifestCaseSha256": canonical_json_sha256(row),
                },
            }
        )
    if missing_cases and not allow_missing:
        raise BlindProductMatrixError(
            "selected local media is missing: " + ", ".join(missing_cases)
        )
    if not frozen_cases:
        raise BlindProductMatrixError("no selected local media is available")

    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "truth-redacted-local-product-matrix",
        "sourceManifest": {
            "path": str(source_path),
            "fileSha256": sha256_file(source_path),
            "libraryId": source.get("libraryId"),
        },
        "selection": {
            "splits": list(normalized_splits),
            "heldOutExplicitlyUnlocked": unlock_held_out,
            "explicitCaseSelection": bool(case_ids),
            "allowMissing": allow_missing,
        },
        "truthPersistencePolicy": {
            "referenceTranscriptPersisted": False,
            "referenceTimelinePersisted": False,
            "expectedAnswerPersisted": False,
            "modelIdentityInReviewPacket": False,
        },
        "counts": {
            "cases": len(frozen_cases),
            "missingCases": len(missing_cases),
        },
        "missingCaseIds": missing_cases,
        "cases": frozen_cases,
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", action="append", choices=_ALLOWED_SPLITS, default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--unlock-held-out", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = build_matrix(
        source_manifest=args.source_manifest,
        audio_root=args.audio_root,
        output_path=args.output,
        splits=args.split or ("development",),
        case_ids=args.case,
        allow_missing=args.allow_missing,
        unlock_held_out=args.unlock_held_out,
    )
    atomic_write_json_no_replace(args.output, matrix)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "caseCount": matrix["counts"]["cases"],
                "missingCaseCount": matrix["counts"]["missingCases"],
                "canonicalSha256": matrix["canonicalSha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
