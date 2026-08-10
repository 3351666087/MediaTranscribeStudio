"""Stage ASCEND held-out media from the public truth-redacted freeze.

This tool deliberately has no scorer-vault argument.  It reads only the
ordinary public freeze, verifies its integrity and redaction policy, selects
the six held-out cases, and publishes a self-contained product-run manifest
under an allowed production input root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from tools.freeze_ascend_code_switch_samples import (  # noqa: E402
    CONFIG,
    DATASET,
    LICENSE,
    REVISION,
    _rename_directory_no_replace,
    assert_truth_redacted,
)


DEFAULT_PUBLIC_MANIFEST = (
    Path("D:/mts-eval/ascend-code-switch-v1/ascend-code-switch-frozen.v1.json")
    if os.name == "nt"
    else Path(
        "/mnt/d/mts-eval/ascend-code-switch-v1/"
        "ascend-code-switch-frozen.v1.json"
    )
)
DEFAULT_STAGE_ROOT = (
    PROJECT_ROOT
    / ".runtime_cache"
    / "sample-library"
    / "ascend-heldout-product-v1"
)
MANIFEST_NAME = "ascend-heldout-product-manifest.v1.json"
ARTIFACT_TYPE = "ascend-code-switch-held-out-product-stage"
_EXPECTED_LIBRARY_ID = "mts-caire-ascend-code-switch-v1"
_EXPECTED_TRUTH_POLICY = {
    "ordinaryManifestContainsTranscript": False,
    "developmentReferencePersistedSeparately": True,
    "developmentReferenceMayEnterReviewerPacket": False,
    "heldOutTruthPersistedOnlyInIsolatedScorerVault": True,
    "heldOutTruthInOrdinaryManifest": False,
    "heldOutTruthInReviewerPacket": False,
    "scorerVaultPublishedBeforeOrdinaryManifest": True,
}


class AscendHeldOutProductStageError(ValueError):
    """Raised when a held-out product stage cannot be built safely."""


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise AscendHeldOutProductStageError(
            f"public manifest contains non-finite JSON number: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AscendHeldOutProductStageError(
                    f"public manifest contains duplicate field: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except AscendHeldOutProductStageError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AscendHeldOutProductStageError(
            f"cannot read public ASCEND manifest: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise AscendHeldOutProductStageError(
            "public ASCEND manifest must contain an object"
        )
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AscendHeldOutProductStageError(f"{field} must be an object")
    return value


def _non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AscendHeldOutProductStageError(f"{field} must be non-empty text")
    return value.strip()


def _load_public_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise AscendHeldOutProductStageError(
            "public ASCEND manifest must not be a symbolic link"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AscendHeldOutProductStageError(
            f"public ASCEND manifest is missing: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise AscendHeldOutProductStageError(
            "public ASCEND manifest is not a regular file"
        )
    value = _load_json_object(resolved)
    if value.get("schemaVersion") != "1.0.0":
        raise AscendHeldOutProductStageError(
            "public ASCEND manifest schemaVersion is unsupported"
        )
    if value.get("artifactType") != "ascend-code-switch-truth-redacted-freeze":
        raise AscendHeldOutProductStageError(
            "source is not the public truth-redacted ASCEND freeze"
        )
    if value.get("libraryId") != _EXPECTED_LIBRARY_ID:
        raise AscendHeldOutProductStageError("ASCEND libraryId changed")
    if value.get("truthPersistencePolicy") != _EXPECTED_TRUTH_POLICY:
        raise AscendHeldOutProductStageError(
            "public ASCEND truth-persistence policy changed"
        )
    publication = _mapping(value.get("publication"), field="publication")
    if publication != {
        "policy": "atomic-directory-no-replace",
        "manifestWrittenLast": True,
    }:
        raise AscendHeldOutProductStageError(
            "public ASCEND freeze is not an immutable completed publication"
        )
    source = _mapping(value.get("source"), field="source")
    expected_source = {
        "dataset": DATASET,
        "revision": REVISION,
        "config": CONFIG,
        "license": LICENSE,
        "homepage": "https://huggingface.co/datasets/CAiRE/ASCEND",
        "attributionPath": "ascend-code-switch/ATTRIBUTION.md",
    }
    if dict(source) != expected_source:
        raise AscendHeldOutProductStageError(
            "public ASCEND source provenance is not pinned"
        )
    declared = value.get("canonicalSha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise AscendHeldOutProductStageError(
            "public ASCEND canonicalSha256 is invalid"
        )
    body = dict(value)
    body.pop("canonicalSha256")
    if declared != canonical_json_sha256(body):
        raise AscendHeldOutProductStageError(
            "public ASCEND manifest canonicalSha256 does not match"
        )
    try:
        assert_truth_redacted(value)
    except ValueError as exc:
        raise AscendHeldOutProductStageError(str(exc)) from exc
    return resolved, value


def _resolve_public_file(
    source_root: Path,
    raw_path: Any,
    *,
    field: str,
) -> Path:
    text = _non_empty_text(raw_path, field=field)
    if "\\" in text:
        raise AscendHeldOutProductStageError(
            f"{field} must use a relative POSIX path"
        )
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise AscendHeldOutProductStageError(
            f"{field} must stay inside the public freeze"
        )
    lexical = source_root.joinpath(*relative.parts)
    if lexical.is_symlink():
        raise AscendHeldOutProductStageError(f"{field} must not be a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise AscendHeldOutProductStageError(
            f"{field} does not resolve inside the public freeze"
        ) from exc
    if not resolved.is_file():
        raise AscendHeldOutProductStageError(f"{field} is not a regular file")
    return resolved


def _verified_media(
    case: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[Path, dict[str, Any]]:
    case_id = _non_empty_text(case.get("id"), field="cases[].id")
    media = _mapping(case.get("media"), field=f"{case_id}.media")
    source = _resolve_public_file(
        source_root,
        media.get("path"),
        field=f"{case_id}.media.path",
    )
    declared_size = media.get("bytes")
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size <= 0
        or source.stat().st_size != declared_size
    ):
        raise AscendHeldOutProductStageError(
            f"{case_id} media byte count does not match"
        )
    declared_sha = media.get("sha256")
    if not isinstance(declared_sha, str) or sha256_file(source) != declared_sha:
        raise AscendHeldOutProductStageError(
            f"{case_id} media sha256 does not match"
        )
    try:
        with wave.open(str(source), "rb") as handle:
            observed = {
                "sampleRateHz": handle.getframerate(),
                "channels": handle.getnchannels(),
                "sampleWidthBytes": handle.getsampwidth(),
                "frameCount": handle.getnframes(),
            }
    except (OSError, EOFError, wave.Error) as exc:
        raise AscendHeldOutProductStageError(
            f"{case_id} media is not a readable WAV"
        ) from exc
    if observed != {
        field: media.get(field)
        for field in (
            "sampleRateHz",
            "channels",
            "sampleWidthBytes",
            "frameCount",
        )
    }:
        raise AscendHeldOutProductStageError(
            f"{case_id} WAV evidence does not match"
        )
    if (
        observed["sampleRateHz"] != 16_000
        or observed["channels"] != 1
        or observed["sampleWidthBytes"] != 2
        or observed["frameCount"] <= 0
    ):
        raise AscendHeldOutProductStageError(
            f"{case_id} WAV is not non-empty mono 16 kHz PCM16"
        )
    return source, dict(media)


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def build_stage(*, public_manifest: Path, stage_root: Path) -> dict[str, Any]:
    """Build and atomically publish one held-out-only product input stage."""

    source_path, public = _load_public_manifest(public_manifest)
    target = stage_root.expanduser().absolute().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_root = source_path.parent.resolve(strict=True)
    if target == source_root or target.is_relative_to(source_root):
        raise AscendHeldOutProductStageError(
            "stage root must be separate from the public freeze"
        )

    rows = public.get("cases")
    if not isinstance(rows, list) or len(rows) != 12:
        raise AscendHeldOutProductStageError(
            "public ASCEND freeze must contain exactly twelve cases"
        )
    held_out = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("evaluationSplit") == "held-out"
    ]
    if len(held_out) != 6:
        raise AscendHeldOutProductStageError(
            "public ASCEND freeze must contain exactly six held-out cases"
        )
    case_ids = [_non_empty_text(row.get("id"), field="cases[].id") for row in held_out]
    if len(set(case_ids)) != len(case_ids):
        raise AscendHeldOutProductStageError("held-out case IDs must be unique")
    for row in held_out:
        case_id = str(row["id"])
        source_key = _mapping(row.get("sourceKey"), field=f"{case_id}.sourceKey")
        if (
            row.get("tuningEligible") is not False
            or row.get("truthAccess") != "isolated-scorer-vault-only"
            or source_key.get("split") != "test"
        ):
            raise AscendHeldOutProductStageError(
                f"{case_id} is not a sealed test-split held-out case"
            )

    temporary_root = Path(
        tempfile.mkdtemp(prefix=".ascend-heldout-product-", dir=target.parent)
    )
    try:
        staged_cases: list[dict[str, Any]] = []
        for row in held_out:
            case_id = str(row["id"])
            source_media, media = _verified_media(row, source_root=source_root)
            target_media = temporary_root / "media" / f"{case_id}.wav"
            _copy_new(source_media, target_media)
            if (
                target_media.stat().st_size != media["bytes"]
                or sha256_file(target_media) != media["sha256"]
            ):
                raise AscendHeldOutProductStageError(
                    f"{case_id} staged media verification failed"
                )
            staged_row = dict(row)
            staged_row["media"] = {
                **media,
                "path": target_media.relative_to(temporary_root).as_posix(),
            }
            staged_cases.append(staged_row)

        attribution = _mapping(public.get("attribution"), field="attribution")
        source_attribution = _resolve_public_file(
            source_root,
            attribution.get("path"),
            field="attribution.path",
        )
        declared_attribution_sha = attribution.get("fileSha256")
        if (
            not isinstance(declared_attribution_sha, str)
            or sha256_file(source_attribution) != declared_attribution_sha
        ):
            raise AscendHeldOutProductStageError(
                "ASCEND attribution sha256 does not match"
            )
        target_attribution = temporary_root / "ATTRIBUTION.md"
        _copy_new(source_attribution, target_attribution)

        source_keys = [
            _mapping(row["sourceKey"], field=f"{row['id']}.sourceKey")
            for row in staged_cases
        ]
        body = {
            "schemaVersion": "1.0.0",
            "artifactType": ARTIFACT_TYPE,
            "libraryId": public["libraryId"],
            "generatedAt": datetime.now(UTC).isoformat(),
            "sourceManifest": {
                "path": str(source_path),
                "fileSha256": sha256_file(source_path),
                "canonicalSha256": public["canonicalSha256"],
                "artifactType": public["artifactType"],
            },
            "source": dict(_mapping(public["source"], field="source")),
            "selection": {
                "evaluationSplit": "held-out",
                "sourceSplit": "test",
                "tuningEligible": False,
                "filterPolicy": "public-truth-redacted-manifest-only",
                "scorerVaultRead": False,
                "speakerCountMode": "auto",
            },
            "truthPersistencePolicy": {
                "sourceManifestTruthRedacted": True,
                "referenceTranscriptPersisted": False,
                "referenceTimelinePersisted": False,
                "scorerTruthRead": False,
                "scorerVaultDependency": False,
                "developmentReferenceCopied": False,
            },
            "attribution": {
                "path": target_attribution.relative_to(temporary_root).as_posix(),
                "fileSha256": sha256_file(target_attribution),
            },
            "coverage": {
                "cases": len(staged_cases),
                "distinctSpeakers": len({row["speaker"] for row in source_keys}),
                "distinctSpeakerSessions": len(
                    {(row["speaker"], row["session"]) for row in source_keys}
                ),
                "distinctTopics": len({row["topic"] for row in staged_cases}),
                "durationBuckets": dict(
                    Counter(str(row["durationBucket"]) for row in staged_cases)
                ),
            },
            "cases": staged_cases,
            "publication": {
                "policy": "atomic-directory-no-replace",
                "manifestWrittenLast": True,
            },
        }
        assert_truth_redacted(body)
        manifest = {**body, "canonicalSha256": canonical_json_sha256(body)}
        staged_manifest = temporary_root / MANIFEST_NAME
        atomic_write_json(staged_manifest, manifest)
        _rename_directory_no_replace(temporary_root, target)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)

    final_manifest = target / MANIFEST_NAME
    if not final_manifest.is_file():
        raise AscendHeldOutProductStageError(
            "ASCEND held-out product stage publication is incomplete"
        )
    return {
        "stageRoot": str(target),
        "manifest": str(final_manifest),
        "manifestFileSha256": sha256_file(final_manifest),
        "manifestCanonicalSha256": manifest["canonicalSha256"],
        "caseCount": len(staged_cases),
        "caseIds": case_ids,
        "scorerVaultRead": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=DEFAULT_PUBLIC_MANIFEST,
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_stage(
        public_manifest=args.public_manifest,
        stage_root=args.stage_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
